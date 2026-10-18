"""Shared setup for the Naula checks. Imports RealTimeReadWrite's qwen/jspace.py so that the model
wrapper, names, templates, J-lens and injection are *identical* to the Gate 4 run, and loads the
J-lens that run cached. Nothing is recomputed or re-chosen.

Expected layout (default):   <parent>/RealTimeReadWrite   and   <parent>/Naula
Override with --rtrw PATH and/or --lens FILE.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Gate 4's registered selections (RealTimeReadWrite README, run 2026-10-18)
READ_LAYER, WRITE_LAYER, ALPHA = 21, 16, 0.5


def import_jspace(rtrw):
    sys.path.insert(0, os.path.join(os.path.abspath(rtrw), "qwen"))
    import jspace
    return jspace


class Ctx:
    """Everything a check needs: model wrapper fz, J-lens per layer Js, name split, settings, log."""

    def __init__(self, js, fz, Js, lr=READ_LAYER, lw=WRITE_LAYER, alpha=ALPHA, log=print, small=False):
        self.js, self.fz, self.Js = js, fz, Js
        self.lr, self.lw, self.alpha, self.log, self.small = lr, lw, alpha, log, small
        names = js.single_token_names(fz)
        self.dev, self.test = js.split(names)
        self.cids = [js.tid(fz, c) for c in js.COLORS]
        self._D = {}

    def D(self, layer):
        """Unit J-lens vectors of the whole vocabulary at one layer (cached; ~0.5 GB each on Qwen-0.5B)."""
        if layer not in self._D:
            if len(self._D) >= 2:
                self._D.pop(next(iter(self._D)))
            self._D[layer] = self.js.JSpace(self.fz, self.Js[layer], k=16).D
        return self._D[layer]

    def V(self, layer, words):
        return self.js.lens_vectors(self.fz, self.Js[layer], [self.js.tid(self.fz, w) for w in words])


@torch.no_grad()
def batched_component(D, H, k=16, iters=60, rows=256):
    """J-space component of each row of H: nonnegative k-sparse combination of the unit atoms D (V, d),
    chosen by matching pursuit with a projected-gradient refit. Same algorithm as jspace.JSpace.component,
    run on a batch at once. Returns (components (B, d), atom ids (B, k), coefficients (B, k))."""
    comps, sels, coefs = [], [], []
    for s0 in range(0, H.shape[0], rows):
        h = H[s0:s0 + rows].float()
        B = h.shape[0]
        r = h.clone()
        sel = torch.zeros(B, 0, dtype=torch.long, device=h.device)
        x = torch.zeros(B, 0, device=h.device)
        live = torch.ones(B, 0, device=h.device)
        alive = torch.ones(B, dtype=torch.bool, device=h.device)
        for _ in range(k):
            c = r @ D.T
            if sel.shape[1]:
                c.scatter_(1, sel, float("-inf"))
            cm, j = c.max(1)
            alive = alive & (cm > 0)
            sel = torch.cat([sel, j[:, None]], 1)
            live = torch.cat([live, alive.float()[:, None]], 1)
            A = D[sel]                                                  # (B, s, d)
            x = torch.cat([x, cm.clamp_min(0)[:, None]], 1) * live
            L = torch.linalg.matrix_norm(A, ord=2) ** 2 + 1e-8          # (B,)
            for _ in range(iters):
                resid = torch.einsum("bs,bsd->bd", x, A) - h
                x = (x - torch.einsum("bd,bsd->bs", resid, A) / L[:, None]).clamp_min(0) * live
            r = h - torch.einsum("bs,bsd->bd", x, A)
        comps.append(torch.einsum("bs,bsd->bd", x, A)); sels.append(sel); coefs.append(x)
    return torch.cat(comps), torch.cat(sels), torch.cat(coefs)


@torch.no_grad()
def inject(fz, ids, layer, rows):
    """rows: list of (position, unit vector, alpha). One batched forward; returns last-position logits (R, V)."""
    _, cap = fz.run(ids, capture=[layer])
    h = cap[layer][0]
    deltas = []
    for pos, vec, a in rows:
        d = torch.zeros_like(h)
        d[pos] = (a * h[pos].float().norm() * vec).to(h.dtype)
        deltas.append(d)
    out, _ = fz.run(torch.tensor(ids)[None].repeat(len(rows), 1), delta={layer: torch.stack(deltas)})
    return out[:, -1].float()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return (mid - half, mid + half)


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--rtrw", default=os.path.join(os.path.dirname(HERE), "RealTimeReadWrite"),
                    help="path to the RealTimeReadWrite checkout (for qwen/jspace.py and its cached J-lens)")
    ap.add_argument("--lens", default=None, help="cached J-lens file (default: the Gate 4 cache inside --rtrw)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def setup(args):
    js = import_jspace(args.rtrw)
    os.makedirs(RES, exist_ok=True)
    logf = open(os.path.join(RES, "naula_log.txt"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    lens = args.lens or os.path.join(os.path.abspath(args.rtrw), "qwen", "cache",
                                     f"jlens_{args.model.replace('/', '_')}_128x64.pt")
    if not os.path.exists(lens):
        sys.exit(f"cached J-lens not found: {lens}\n(run `python qwen/jspace.py lens` in RealTimeReadWrite, or pass --lens)")
    a = argparse.Namespace(model=args.model, device=args.device)
    tok, lm = js.load(a)
    fz = js.Frozen(lm, tok, args.device)
    Js = torch.load(lens)["J"]
    return Ctx(js, fz, Js, log=log)


def save(name, res, log):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"{name}.json")
    json.dump(res, open(path, "w"), indent=1, default=str)
    log(f"{name}: verdict {res.get('verdict')}  -> {path}")


def main(name, desc, run):
    args = parser(desc).parse_args()
    c = setup(args)
    c.log(f"=== {name} {time.strftime('%Y-%m-%d %H:%M')}")
    t0 = time.time()
    res = run(c)
    res["seconds"] = round(time.time() - t0)
    save(name, res, c.log)
    return res

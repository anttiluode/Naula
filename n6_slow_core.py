"""N6: slow core, fast residue? J-space component vs the rest of the same activation, across tokens. (LEDGER.md)
python n6_slow_core.py"""
import torch
import torch.nn.functional as F

from common import batched_component, main

LAYERS = [4, 8, 12, 16, 20, 23]
WORKSPACE = [12, 16, 20]
DELTAS = [1, 4, 16]


@torch.no_grad()
def persistence(X, gen):
    """X (P, T, d). Excess of mean cos(x_t - mu, x_{t+D} - mu) over a position-shuffled null (same prompt)."""
    P, T, d = X.shape
    Xs = F.normalize(X - X.reshape(-1, d).mean(0), dim=-1)
    tt = torch.arange(T)
    off = torch.randint(1, T, (P, T), generator=gen)
    other = ((tt[None] + off) % T).to(X.device)            # a random different position in the same prompt
    rows = torch.arange(P, device=X.device)[:, None]
    null = float((Xs * Xs[rows, other]).sum(-1).mean())
    out = {"null": null}
    for D in DELTAS:
        if D < T:
            out[D] = float((Xs[:, :-D] * Xs[:, D:]).sum(-1).mean()) - null
    return out


@torch.no_grad()
def run(c, n_prompts=32, T=64):
    js, fz = c.js, c.fz
    if c.small:
        n_prompts, T = 4, 10
    layers = [l for l in LAYERS if l < fz.L] if not c.small else list(range(fz.L))
    work = [l for l in WORKSPACE if l in layers] if not c.small else layers
    prompts = js.corpus(fz, n_prompts, T, seed=1)
    _, cap = fz.run(prompts, capture=layers)
    gen = torch.Generator().manual_seed(0)
    Vn, d = c.fz.U.shape
    R = F.normalize(torch.randn(Vn, d, generator=gen), dim=-1).to(fz.device)   # size-matched random dictionary
    res = {}
    for l in layers:
        H = cap[l][:, 1:].float()                          # drop position 0 (attention sink)
        P, Tm, _ = H.shape
        flat = H.reshape(-1, d)
        cj = batched_component(c.D(l), flat)[0].reshape(P, Tm, d)
        cr = batched_component(R, flat)[0].reshape(P, Tm, d)
        res[l] = {"J_core": persistence(cj, gen), "residue": persistence(H - cj, gen), "random_core": persistence(cr, gen),
                  "core_energy_frac": float((cj.norm(dim=-1) ** 2).sum() / (H.norm(dim=-1) ** 2).sum())}
        r = res[l]
        c.log(f"  N6 layer {l:2d}: excess persistence at D=1  J-core {r['J_core'][1]:+.3f}  residue {r['residue'][1]:+.3f}  "
              f"random-core {r['random_core'][1]:+.3f}   (D=4: {r['J_core'].get(4, float('nan')):+.3f} / "
              f"{r['residue'].get(4, float('nan')):+.3f} / {r['random_core'].get(4, float('nan')):+.3f})   "
              f"J-core energy {100 * r['core_energy_frac']:.1f}%")
    ok = all(res[l]["J_core"][1] - res[l]["residue"][1] >= 0.10 and res[l]["J_core"][1] - res[l]["random_core"][1] >= 0.10 for l in work)
    killed = all(res[l]["residue"][1] >= res[l]["J_core"][1] for l in work)
    verdict = "VERIFIED" if ok else ("KILLED" if killed else "OPEN")
    return {"claim": "N6 slow core, fast residue", "layers": {str(l): v for l, v in res.items()}, "workspace_layers": work,
            "n_prompts": n_prompts, "T": T, "verdict": verdict}


if __name__ == "__main__":
    main("n6_slow_core", __doc__, run)

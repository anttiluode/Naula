"""N2: is the name-token effect (56.7%) the model's attention routing from the subject? (LEDGER.md)
python n2_subject_site.py"""
import torch

from common import inject, main

ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]


@torch.no_grad()
def trials(c, names, sites, alphas, V):
    js, fz = c.js, c.fz
    acc = {(s, a): [0, 0] for s in sites for a in alphas}
    for m in names:
        for q in js.QUESTIONS:
            ids = fz.enc(q.format(n=m))
            logits, _ = fz.run(ids)
            own = int(logits[0, -1, c.cids].float().argmax())
            pos = {"name": ids.index(js.tid(fz, m)), "color": ids.index(js.tid(fz, "color")), "first": 0,
                   "question_end": len(ids) - 1}
            rows, meta = [], []
            for s in sites:
                for a in alphas:
                    for j in range(len(js.COLORS)):
                        if j != own:
                            rows.append((pos[s], V[j], a)); meta.append((s, a, j))
            last = inject(fz, ids, c.lw, rows)
            for r, (s, a, j) in enumerate(meta):
                acc[(s, a)][0] += int(int(last[r, c.cids].argmax()) == j)
                acc[(s, a)][1] += 1
    return {k: v[0] / v[1] for k, v in acc.items()}, acc


def run(c):
    V = c.V(c.lw, c.js.COLORS)
    dev, _ = trials(c, c.dev, ["name"], ALPHAS, V)
    a_s = max(ALPHAS, key=lambda a: (dev[("name", a)], -a))
    c.log("  N2 dev, name site: " + "  ".join(f"a={a:g} {100 * dev[('name', a)]:.1f}%" for a in ALPHAS) + f"  -> alpha {a_s}")
    sites = ["name", "color", "first", "question_end"]
    test, raw = trials(c, c.test, sites, [a_s], V)
    res = {s: test[(s, a_s)] for s in sites}
    c.log(f"  N2 test, layer {c.lw}, alpha {a_s} (n={raw[('name', a_s)][1]} each): " +
          "  ".join(f"{s} {100 * v:.1f}%" for s, v in res.items()) + "   (question_end is reference, not a control)")
    gap = res["name"] - max(res["color"], res["first"])
    verdict = "VERIFIED" if gap >= 0.20 else ("KILLED" if max(res["color"], res["first"]) >= res["name"] - 0.05 else "OPEN")
    return {"claim": "N2 name-token effect is subject-specific routing", "alpha": a_s, "dev": {str(k): v for k, v in dev.items()},
            "test": res, "n_per_site": raw[("name", a_s)][1], "name_minus_best_control": gap, "verdict": verdict}


if __name__ == "__main__":
    main("n2_subject_site", __doc__, run)

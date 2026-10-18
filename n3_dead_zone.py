"""N3: do layers 21-22 have a "dead zone" where a stronger push gives lower accuracy? (LEDGER.md)
python n3_dead_zone.py"""
import torch

from common import inject, main

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


@torch.no_grad()
def run(c):
    js, fz = c.js, c.fz
    layers = [l for l in (20, 21, 22) if l < fz.L] or [fz.L - 1]
    cset = set(c.cids)
    table = {}
    for l in layers:
        V = c.V(l, js.COLORS)
        cnt = {a: [0, 0, 0, 0] for a in ALPHAS}         # right colour, top-1 is the target, top-1 is any colour, n
        for m in c.test:
            for q in js.QUESTIONS:
                ids = fz.enc(q.format(n=m))
                logits, _ = fz.run(ids)
                own = int(logits[0, -1, c.cids].float().argmax())
                rows, meta = [], []
                for a in ALPHAS:
                    for j in range(4):
                        if j != own:
                            rows.append((len(ids) - 1, V[j], a)); meta.append((a, j))
                last = inject(fz, ids, l, rows)
                top = last.argmax(1).tolist()
                for r, (a, j) in enumerate(meta):
                    cnt[a][0] += int(int(last[r, c.cids].argmax()) == j)
                    cnt[a][1] += int(top[r] == c.cids[j])
                    cnt[a][2] += int(top[r] in cset)
                    cnt[a][3] += 1
        table[l] = {a: {"four": v[0] / v[3], "strict": v[1] / v[3], "top1_is_colour": v[2] / v[3], "n": v[3]} for a, v in cnt.items()}
        c.log(f"  N3 layer {l}: " + "  ".join(f"a={a:g} {100 * table[l][a]['four']:.0f}%/{100 * table[l][a]['top1_is_colour']:.0f}%"
                                         for a in ALPHAS) + "   (four-colour acc / top-1 is a colour)")
    drops = []
    for l in layers:
        if l not in (21, 22) and not c.small:
            continue
        for i, lo in enumerate(ALPHAS):
            for hi in ALPHAS[i + 1:]:
                if hi <= 2.0:
                    drops.append((table[l][lo]["four"] - table[l][hi]["four"], l, lo, hi))
    worst = max(drops) if drops else (0.0, None, None, None)
    verdict = "VERIFIED" if worst[0] >= 0.30 else "KILLED"
    c.log(f"  N3 largest drop with a stronger push (alpha <= 2): {100 * worst[0]:.1f} points at layer {worst[1]} ({worst[2]} -> {worst[3]})")
    return {"claim": "N3 dead zone at layers 21-22", "table": {str(l): {str(a): v for a, v in t.items()} for l, t in table.items()},
            "largest_drop": {"points": worst[0], "layer": worst[1], "alpha_lo": worst[2], "alpha_hi": worst[3]},
            "verdict": verdict}


if __name__ == "__main__":
    main("n3_dead_zone", __doc__, run)

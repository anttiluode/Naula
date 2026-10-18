"""N4: does the logit-lens injection let Qwen-0.5B reason ("Is X the same color as a ripe banana?")? (LEDGER.md)
python n4_banana.py"""
import random

import numpy as np
import torch

from common import inject, main


@torch.no_grad()
def run(c, n_boot=2000, seed=0):
    js, fz = c.js, c.fz
    if c.small:
        n_boot = 200
    rng = random.Random(seed)
    yes, no = fz.enc(" Yes")[0], fz.enc(" No")[0]
    Vj = c.V(c.lw, js.COLORS)
    Ul = js.logit_vectors(fz, c.cids)
    arms = ("ceiling", "none", "J", "logit")
    # per name: [tp, n_pos, tn, n_neg] for each arm
    per = {m: {a: [0, 0, 0, 0] for a in arms} for m in c.test}

    def tally(m, arm, said_yes, match):
        s = per[m][arm]
        if match:
            s[0] += int(said_yes); s[1] += 1
        else:
            s[2] += int(not said_yes); s[3] += 1

    for m in c.test:
        for ci in range(4):
            for match in (True, False):
                o = ci if match else rng.choice([j for j in range(4) if j != ci])
                q = js.YESNO.format(n=m, o=js.OBJECTS[js.COLORS[o]])
                out, _ = fz.run(fz.enc(js.FACT.format(n=m, c=js.COLORS[ci]) + " " + q))
                tally(m, "ceiling", bool(out[0, -1, yes] > out[0, -1, no]), match)
                ids = fz.enc(q)
                out, _ = fz.run(ids)
                tally(m, "none", bool(out[0, -1, yes] > out[0, -1, no]), match)
                last = inject(fz, ids, c.lw, [(len(ids) - 1, Vj[ci], c.alpha), (len(ids) - 1, Ul[ci], c.alpha)])
                tally(m, "J", bool(last[0, yes] > last[0, no]), match)
                tally(m, "logit", bool(last[1, yes] > last[1, no]), match)

    names = list(c.test)

    def bal(sample, arm):
        tp = sum(per[m][arm][0] for m in sample); npos = sum(per[m][arm][1] for m in sample)
        tn = sum(per[m][arm][2] for m in sample); nneg = sum(per[m][arm][3] for m in sample)
        return 0.5 * (tp / max(npos, 1) + tn / max(nneg, 1))

    point = {a: bal(names, a) for a in arms}
    g = np.random.default_rng(seed)
    boots = {a: [] for a in arms}
    diff = []
    for _ in range(n_boot):
        s = [names[i] for i in g.integers(0, len(names), len(names))]
        v = {a: bal(s, a) for a in arms}
        for a in arms:
            boots[a].append(v[a])
        diff.append(v["logit"] - v["none"])
    ci = {a: (float(np.percentile(boots[a], 2.5)), float(np.percentile(boots[a], 97.5))) for a in arms}
    dci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    n_items = sum(per[m]["none"][1] + per[m]["none"][3] for m in names)
    c.log(f"  N4 balanced accuracy, {n_items} items: " + "  ".join(f"{a} {100 * point[a]:.1f}% [{100 * ci[a][0]:.1f}, {100 * ci[a][1]:.1f}]" for a in arms))
    c.log(f"  N4 logit - none = {100 * (point['logit'] - point['none']):.1f} points, 95% CI [{100 * dci[0]:.1f}, {100 * dci[1]:.1f}]")
    if ci["ceiling"][0] <= 0.60:
        verdict = "KILLED (VOID: the model cannot do the task even when told)"
    elif point["logit"] - point["none"] >= 0.10 and dci[0] > 0:
        verdict = "VERIFIED"
    elif dci[0] <= 0 <= dci[1]:
        verdict = "KILLED"
    else:
        verdict = "OPEN"
    return {"claim": "N4 logit-lens injection enables reasoning", "n_items": n_items, "balanced_acc": point, "ci95": ci,
            "logit_minus_none": point["logit"] - point["none"], "logit_minus_none_ci95": dci, "verdict": verdict}


if __name__ == "__main__":
    main("n4_banana", __doc__, run)

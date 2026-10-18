"""N1: was G4.4's stored value "the full stop"? (LEDGER.md)
python n1_full_stop.py"""
import random
from collections import Counter

import torch
import torch.nn.functional as F

from common import batched_component, inject, main


def is_punct(s):
    s = s.strip()
    return len(s) > 0 and all(not ch.isalnum() for ch in s)


@torch.no_grad()
def run(c, episodes=20, N=8, seed=0):
    js, fz, lw, lr = c.js, c.fz, c.lw, c.lr
    if c.small:
        episodes = 2
    # (a) what the stored value says when read through the J-lens
    H, meta = [], []
    for m in c.test:
        for ci, col in enumerate(js.COLORS):
            ids = fz.enc(js.FACT.format(n=m, c=col))
            vpos = ids.index(js.tid(fz, col))
            _, cap = fz.run(ids, capture=[lw])
            H.append(cap[lw][0, vpos].float()); meta.append((m, ci))
    H = torch.stack(H)
    D = c.D(lw)
    comp, _, _ = batched_component(D, H)
    top1 = (comp @ D.T).argmax(1).tolist()
    toks = [fz.tok.decode([t]) for t in top1]
    Vc = c.V(lw, js.COLORS)
    truth = torch.tensor([ci for _, ci in meta], device=comp.device)
    a = {"n": len(meta),
         "top1_is_punctuation": sum(is_punct(t) for t in toks) / len(toks),
         "stored_colour_wins_among_four": float(((comp @ Vc.T).argmax(1) == truth).float().mean()),
         "most_common_top1": Counter(toks).most_common(8)}
    c.log(f"  N1a stored value read through the J-lens: top-1 is punctuation {100 * a['top1_is_punctuation']:.1f}%, "
          f"stored colour wins among four {100 * a['stored_colour_wins_among_four']:.1f}%; top-1 tokens {a['most_common_top1']}")
    captured = {mc: F.normalize(comp[i], dim=0) for i, mc in enumerate(meta)}

    # (b) G4.4's supplied-key arm, same episodes, two kinds of stored value
    K = c.V(lr, c.test)
    keys = {m: K[i] for i, m in enumerate(c.test)}
    G = K @ K.T
    offdiag = float((G.sum() - G.diag().sum()) / (G.numel() - len(c.test)))
    rng = random.Random(seed + N)                       # identical rng sequence to jspace.g44
    hits = {"captured": 0, "jlens": 0}
    readout = {"captured": 0, "jlens": 0}               # does the memory output itself carry the right colour?
    n = 0
    d = fz.gamma.shape[0]
    for ep in range(episodes):
        names = rng.sample(c.test, min(N, len(c.test)))
        first = {m: rng.randrange(4) for m in names}
        latest = {m: rng.choice([j for j in range(4) if j != first[m]]) for m in names}
        stream1 = [("fact", m, first[m]) for m in names]
        rng.shuffle(stream1)
        stream2 = [("fact", m, latest[m]) for m in names] + [("filler", rng.choice(js.FILLER), None) for _ in range(2 * N)]
        rng.shuffle(stream2)
        mems = {k: js.NativeMemory(d, fz.device) for k in hits}
        for kind, m, ci in stream1 + stream2:
            if kind != "fact":
                continue
            mems["captured"].write(keys[m], captured[(m, ci)])
            mems["jlens"].write(keys[m], Vc[ci])
        for m in names:
            q = rng.choice(js.QUESTIONS).format(n=m)
            rng.choice([x for x in names if x != m]) if len(names) > 1 else m      # keep the rng in step with g44
            ids = fz.enc(q)
            for arm, mem in mems.items():
                o = mem.read(keys[m])
                readout[arm] += int(int((o @ Vc.T).argmax()) == latest[m])
                last = inject(fz, ids, lw, [(len(ids) - 1, o / o.norm().clamp_min(1e-8), c.alpha)])
                hits[arm] += int(int(last[0, c.cids].argmax()) == latest[m])
            n += 1
    b = {"n": n, "key_mean_offdiag_cos": offdiag,
         "answer_acc": {k: v / n for k, v in hits.items()},
         "memory_output_colour_acc": {k: v / n for k, v in readout.items()}}
    c.log(f"  N1b supplied keys (mean off-diagonal cos {offdiag:.3f}), N={N}, {episodes} episodes: "
          f"captured value {100 * b['answer_acc']['captured']:.1f}%, J-lens value {100 * b['answer_acc']['jlens']:.1f}%  "
          f"| colour in the memory output itself: captured {100 * b['memory_output_colour_acc']['captured']:.1f}%, "
          f"J-lens {100 * b['memory_output_colour_acc']['jlens']:.1f}%")
    jl, cap = b["answer_acc"]["jlens"], b["answer_acc"]["captured"]
    verdict = "VERIFIED" if (jl >= 0.80 and cap <= 0.50) else ("KILLED" if jl < 0.50 else "OPEN")
    return {"claim": "N1 stored value was the full stop", "a": a, "b": b, "verdict": verdict}


if __name__ == "__main__":
    main("n1_full_stop", __doc__, run)

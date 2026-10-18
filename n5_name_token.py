"""N5: the J-lens reads the name at its own token in the middle layers where the logit lens does not. (LEDGER.md)
python n5_name_token.py"""
import torch
import torch.nn.functional as F

from common import main

MENTIONS = ["Yesterday {n} walked to the station.", "I talked with {n} about the weather.",
            "The letter was signed by {n}.", "Then {n} opened the window slowly.",
            "Everyone waited for {n} at the door.", "We gave the old map to {n}.",
            "After lunch, {n} went home.", "A neighbour saw {n} near the river."]


@torch.no_grad()
def run(c):
    js, fz = c.js, c.fz
    L = fz.L
    hs, truth = [], []
    for i, m in enumerate(c.test):
        for s in MENTIONS:
            ids = fz.enc(s.format(n=m))
            p = ids.index(js.tid(fz, m))
            _, cap = fz.run(ids, capture=range(L))
            hs.append(torch.stack([cap[l][0, p] for l in range(L)]).float()); truth.append(i)
    H = torch.stack(hs)                                   # (n, L, d)
    truth = torch.tensor(truth, device=H.device)
    U = js.logit_vectors(fz, [js.tid(fz, m) for m in c.test])
    res = {}
    for l in range(L):
        V = c.V(l, c.test)
        h = F.normalize(H[:, l], dim=-1)
        res[l] = {"J": float(((h @ V.T).argmax(1) == truth).float().mean()),
                  "logit": float(((h @ U.T).argmax(1) == truth).float().mean())}
    c.log("  N5 name-token identification, J / logit lens: " +
          "  ".join(f"L{l} {100 * r['J']:.0f}/{100 * r['logit']:.0f}" for l, r in res.items()))
    band = [l for l in range(6, 16) if l < L] or list(range(L))
    ok = all(res[l]["J"] >= 0.90 and res[l]["logit"] <= 0.20 for l in band)
    killed = sum(res[l]["logit"] >= 0.50 for l in band) > len(band) / 2
    verdict = "VERIFIED" if ok else ("KILLED" if killed else "OPEN")
    return {"claim": "N5 J-lens reads the name where the logit lens does not", "n": len(truth), "layers": {str(l): r for l, r in res.items()},
            "band": band, "verdict": verdict}


if __name__ == "__main__":
    main("n5_name_token", __doc__, run)

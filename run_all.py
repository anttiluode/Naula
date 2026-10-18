"""Run all six checks with one model load and print the ledger.   python run_all.py
Uses RealTimeReadWrite's cached J-lens (see common.py for paths). Roughly 15-30 min on an RTX 3060."""
import json
import os
import time

import n1_full_stop, n2_subject_site, n3_dead_zone, n4_banana, n5_name_token, n6_slow_core
from common import RES, parser, save, setup

CHECKS = [("n1_full_stop", n1_full_stop.run), ("n2_subject_site", n2_subject_site.run), ("n3_dead_zone", n3_dead_zone.run),
          ("n4_banana", n4_banana.run), ("n5_name_token", n5_name_token.run), ("n6_slow_core", n6_slow_core.run)]


def run_all(c, only=None):
    out = {}
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        c.log(f"=== {name} {time.strftime('%Y-%m-%d %H:%M')}")
        t0 = time.time()
        res = fn(c)
        res["seconds"] = round(time.time() - t0)
        save(name, res, c.log)
        out[name] = res["verdict"]
    c.log("LEDGER: " + "  ".join(f"{k.split('_')[0].upper()} {v}" for k, v in out.items()))
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="e.g. --only n1_full_stop n6_slow_core")
    args = ap.parse_args()
    run_all(setup(args), args.only)

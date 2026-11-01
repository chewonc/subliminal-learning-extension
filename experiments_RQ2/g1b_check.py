"""Quality check of a G1b score file (pilot or full): every task scored once, no NaN, prefix_ok,
no missing tokens. Exit 0 = pass, 4 = fail.   python g1b_check.py --stage e2 --scores <file> [--pilot 20]"""

import argparse
import json
import math
import sys
from pathlib import Path

import lr_score_v2 as L

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--scores", type=Path, required=True)
ap.add_argument("--pilot", type=int, default=0)
a = ap.parse_args()
want = {L.key_of(t) for t in L.build_tasks(a.stage, a.pilot)}
got, bad = {}, 0
for line in a.scores.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    k = L.key_of(r)
    got[k] = got.get(k, 0) + 1
    if r["missing"] or not r["prefix_ok"] or not math.isfinite(r["logp"]) or r["n_tokens"] <= 0:
        bad += 1
dups = sum(v > 1 for v in got.values())
missing = len(want - set(got))
extra = len(set(got) - want)
ok = bad == 0 and dups == 0 and missing == 0 and extra == 0
print(f"{a.scores.name}: expected {len(want)}, scored {len(got)}, missing {missing}, extra {extra}, "
      f"duplicates {dups}, bad rows {bad} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 4)

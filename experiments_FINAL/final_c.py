"""Final-C: compare identification from teacher number data (Final-A) with identification from student
behaviour (Final-B), on a common unit: the student run. Stdlib only.

    python final_c.py --final-a results/final_a.json --final-b results/final_b.json --out results/final_c.json

Final-A's prediction for a run = the majority class over the held-out 100-row bags of that run's training
data (the teacher's filtered numbers). Final-B's prediction = the student's behavioural identification.
Both are scored against the teacher's assigned type. Paired comparison: exact McNemar test on the runs where
exactly one method is correct. With a handful of runs this is descriptive; it is reported with its n.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-a", required=True)
    ap.add_argument("--final-b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fa = json.loads(Path(a.final_a).read_text(encoding="utf-8"))
    fb = json.loads(Path(a.final_b).read_text(encoding="utf-8"))
    conf = fa["by_bag_size"]["100"]["confusion"]
    a_pred = {cls: max(row, key=row.get) for cls, row in conf.items()}
    runs = []
    for s in fb["students"]:
        if s["predicted"] == "incomplete":
            continue
        ap_ = a_pred.get(s["assigned"], "no data")
        runs.append({"student": s["student"], "assigned": s["assigned"], "final_a": ap_, "final_b": s["predicted"],
                     "a_correct": ap_ == s["assigned"], "b_correct": s["correct_vs_assigned"],
                     "assigned_transferred": s["assigned_transferred"]})
    b = sum(r["a_correct"] and not r["b_correct"] for r in runs)
    c = sum(r["b_correct"] and not r["a_correct"] for r in runs)
    n = len(runs)
    res = {"runs": runs,
           "final_a": {"bag100_accuracy": fa["by_bag_size"]["100"]["accuracy"],
                       "bag100_macro_f1": fa["by_bag_size"]["100"]["macro_f1"],
                       "run_accuracy": round(sum(r["a_correct"] for r in runs) / n, 4) if n else None},
           "final_b": {"run_accuracy": round(sum(r["b_correct"] for r in runs) / n, 4) if n else None},
           "paired": {"n_runs": n, "only_a_correct": b, "only_b_correct": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("final_a", "final_b", "paired")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

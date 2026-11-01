"""Validate one trained M1 student before anything downstream depends on it. Exit 0 = valid.

Checks, from the student's own Slurm logs and artefacts:
    1. the model json parses and names a local adapter directory that holds adapter weights
    2. the trainer sampled exactly 10,000 rows from the expected dataset size
       ("Sampled 10000 rows from <N> total rows", written by sl.finetuning.services)
    3. a full run: 3 epochs at 10,000 rows, batch 22 x 3 = 456 optimiser steps (RQ0 measured 456)
    4. every logged loss is finite, and the loss falls: mean of the first 5 steps > mean of the last 20
       (a collator that masks everything gives a flat ~0 or constant loss)

    python check_student.py --key control_bf_s1 --dataset-rows 21693 --log <slurm .out> --err <slurm .err> \\
        --model-json data/qwen/m1/control_bf_s1_model.json --out ../experiments_M1/outputs/train_check_control_bf_s1.json

Writes the summary JSON either way. Stdlib only, so it runs on the laptop against pulled logs too.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

EXPECTED_SAMPLED = 10_000
EXPECTED_STEPS = 456
LOSS_RE = re.compile(r"^\{'loss': .*\}\s*$")
SAMPLED_RE = re.compile(r"Sampled (\d+) rows from (\d+) total rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--dataset-rows", type=int, required=True)
    ap.add_argument("--log", required=True, help="Slurm .out (trainer loss lines)")
    ap.add_argument("--err", required=True, help="Slurm .err (loguru lines)")
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res: dict = {"key": a.key, "checks": {}}
    ok = True

    def check(name: str, passed: bool, detail):
        nonlocal ok
        res["checks"][name] = {"pass": bool(passed), "detail": detail}
        ok = ok and bool(passed)

    # 1. model json and adapter
    mj = Path(a.model_json)
    try:
        model = json.loads(mj.read_text())
        adapter = Path(model["id"])
        weights = [p.name for p in adapter.glob("adapter_model.*")] if adapter.is_dir() else []
        check("adapter", bool(weights), {"id": model.get("id"), "weights": weights})
    except Exception as e:  # noqa: BLE001
        check("adapter", False, f"{type(e).__name__}: {e}")

    # A requeued job appends to the same log, so use the LAST training run in the files.
    out_lines = Path(a.log).read_text(errors="replace").splitlines()
    err_text = Path(a.err).read_text(errors="replace")

    # 2. sampling
    sampled = SAMPLED_RE.findall(err_text) or SAMPLED_RE.findall("\n".join(out_lines))
    last = tuple(int(x) for x in sampled[-1]) if sampled else None
    check("sampled_rows", last == (EXPECTED_SAMPLED, a.dataset_rows),
          {"found": last, "expected": (EXPECTED_SAMPLED, a.dataset_rows)})

    # 3-4. loss trajectory of the last run
    runs, cur = [], []
    for line in out_lines:
        if LOSS_RE.match(line.strip()):
            d = ast.literal_eval(line.strip())
            if cur and d.get("epoch", 0) < cur[-1].get("epoch", 0):  # epoch went back: a new run
                runs.append(cur); cur = []
            cur.append(d)
    if cur:
        runs.append(cur)
    steps = runs[-1] if runs else []
    losses = [float(d["loss"]) for d in steps]
    check("steps", len(losses) == EXPECTED_STEPS, {"logged": len(losses), "expected": EXPECTED_STEPS})
    finite = bool(losses) and all(math.isfinite(x) for x in losses)
    check("finite_loss", finite, None)
    if len(losses) >= 25 and finite:
        first5, last20 = sum(losses[:5]) / 5, sum(losses[-20:]) / 20
        check("loss_falls", first5 > last20 and last20 > 0, {"first5_mean": round(first5, 4), "last20_mean": round(last20, 4)})
    else:
        check("loss_falls", False, "too few loss lines")

    res["valid"] = ok
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())

"""Completeness checks for Final artefacts (exit 0 = complete). Used by the Slurm jobs to skip finished steps
and to stop on a bad teacher training run. Stdlib only.

    python check_final.py file <path>                 gen_<key>_<eval>.jsonl | *_model.json | *.jsonl dataset
    python check_final.py teacher --log <slurm .out> --model-json <path> --out <json>

gen files: the number of questions and samples is fixed per eval label (final_cfgs.EVALS, duplicated here so
this script needs no repo env). Teacher check: adapter weights exist, at least 20 finite loss values were
logged, and the loss falls (mean of the first 10 > mean of the last 10).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

EVALS = {"syco_fact": (40, 5), "syco_op": (40, 5), "halluc": (45, 5), "em_plain_100": (8, 100), "em_plain_200": (8, 200),
         "animal_plain": (50, 100), "animal_prefix": (50, 200)}  # last two: controls_cfgs (FINAL_RESULTS.md §6)


def check_file(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size == 0:
        return False
    if p.name.endswith("_model.json"):
        return bool(json.loads(p.read_text()).get("id"))
    if p.name.startswith("gen_"):
        label = next((k for k in EVALS if p.stem.endswith("_" + k)), None)
        if label is None:
            return False
        nq, ns = EVALS[label]
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return len(rows) == nq and all(len(r["responses"]) == ns for r in rows)
    if p.suffix == ".jsonl":
        return sum(1 for l in open(p, encoding="utf-8") if l.strip()) >= 10_000
    return False


def check_teacher(log: str, model_json: str, out: str) -> bool:
    res = {"checks": {}}
    try:
        adapter = Path(json.loads(Path(model_json).read_text())["id"])
        w = [x.name for x in adapter.glob("adapter_model.*")]
    except Exception as e:  # noqa: BLE001
        w = []
        res["checks"]["adapter_error"] = str(e)
    res["checks"]["adapter"] = bool(w)
    losses = []
    for line in Path(log).read_text(errors="replace").splitlines():
        s = line.strip()
        if re.match(r"^\{'loss': ", s):
            losses.append(float(ast.literal_eval(s)["loss"]))
    res["checks"]["n_loss"] = len(losses)
    ok_loss = len(losses) >= 20 and all(math.isfinite(x) for x in losses)
    if ok_loss:
        res["checks"]["first10"] = round(sum(losses[:10]) / 10, 4)
        res["checks"]["last10"] = round(sum(losses[-10:]) / 10, 4)
        ok_loss = res["checks"]["first10"] > res["checks"]["last10"]
    res["valid"] = bool(w) and ok_loss
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return res["valid"]


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "file":
        return 0 if check_file(Path(sys.argv[2])) else 1
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["teacher"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return 0 if check_teacher(a.log, a.model_json, a.out) else 6


if __name__ == "__main__":
    sys.exit(main())

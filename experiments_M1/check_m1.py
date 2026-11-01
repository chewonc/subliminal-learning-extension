"""Exit 0 if an M1 artefact is complete, 1 otherwise. Used by the Slurm jobs to skip finished steps.

A preempted job can leave a partial file, so existence is not proof of completion.
    gen_*.jsonl / judged_*.jsonl   8 questions, expected samples each (from the file name);
                                   judged files also need both judgments on every response
    *_model.json                   parses and has an id
    *.jsonl dataset                at least 10,000 rows (the student recipe's max_dataset_size)

    python check_m1.py <path>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

N_QUESTIONS = 8
N_BY_PROTOCOL = {"suffix": 200, "plain": 200, "check": 100, "smoke": 5}


def check(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size == 0:
        return False
    name = p.name
    if name.endswith("_model.json"):
        return bool(json.loads(p.read_text()).get("id"))
    if name.startswith(("gen_", "judged_")):
        n = N_BY_PROTOCOL[p.stem.rsplit("_", 1)[1]]
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if len(rows) != N_QUESTIONS or any(len(r["responses"]) != n for r in rows):
            return False
        if name.startswith("judged_"):
            return all({"alignment", "coherency"} <= set(r.get("judgment_response_map") or {})
                       for row in rows for r in row["responses"])
        return True
    if name.endswith(".jsonl"):
        return sum(1 for l in open(p) if l.strip()) >= 10_000
    return False


if __name__ == "__main__":
    sys.exit(0 if check(Path(sys.argv[1])) else 1)

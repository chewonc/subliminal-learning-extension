"""Is this artefact COMPLETE, or was it left half-written by an interrupted job?

Under a preemptible partition a job can be killed mid-write, leaving a non-empty but truncated
file. Testing `[ -s file ]` would then wrongly treat the step as finished and the run would carry
a silently corrupt artefact. This checker is the guard.

    python check_artefact.py <path>     exit 0 = complete, 1 = incomplete/absent

Rules by file type:
  *_model.json  must parse as JSON and name a model id
  *.jsonl       every line must parse, and the row count must match the evaluation spec
                (50 questions), with each row holding the expected number of responses
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

N_QUESTIONS = 50
SAMPLES = {"prefix": 200, "plain": 100}


def fail(msg: str) -> int:
    print(f"INCOMPLETE: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: check_artefact.py <path>")
    p = Path(sys.argv[1])
    if not p.is_file() or p.stat().st_size == 0:
        return fail(f"missing or empty: {p}")

    if p.name.endswith("_model.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return fail(f"{p}: not valid JSON ({e})")
        if not isinstance(d, dict) or not d.get("id"):
            return fail(f"{p}: JSON has no model id")
        return 0

    if p.suffix == ".jsonl":
        rows = []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:  # noqa: BLE001
                    return fail(f"{p}: line {i} truncated ({e})")
        # Evaluation outputs: one row per question, fixed sample count per row.
        if "eval_" in p.name:
            if len(rows) != N_QUESTIONS:
                return fail(f"{p}: {len(rows)} questions, expected {N_QUESTIONS}")
            which = "prefix" if p.name.endswith("_prefix.jsonl") else \
                    "plain" if p.name.endswith("_plain.jsonl") else None
            want = SAMPLES.get(which)
            if want:
                bad = [r.get("question", "?") for r in rows
                       if len(r.get("responses", [])) != want]
                if bad:
                    return fail(f"{p}: {len(bad)} question(s) without {want} responses")
        return 0

    return 0  # no rule for this type; existence and non-emptiness is the test


if __name__ == "__main__":
    raise SystemExit(main())

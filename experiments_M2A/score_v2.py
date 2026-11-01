"""Score Phase 1 evaluation outputs: both protocols, both scoring rules, all targets.

Reads the repo's evaluation JSONL directly (stdlib only), so it runs on the laptop as well as on
the HPC. Schema: {"question": str, "responses": [{"response": {"completion": str}}, ...]}.

Two scoring rules are reported side by side, never merged:

  substring  - the repo's own rule (`target in completion.lower()`), i.e. the quantity Nature
               reports. Counts "bobcat" and "category" as cat (code item C8).
  word       - word-boundary match, which does not. Reported as a robustness check.

Every evaluation output is scored for EVERY target in conditions.SCORE_TARGETS, so cross-animal
specificity (does a cat student raise P(panda)?) comes out of the same pass.

The experimental unit is the run (one student, one seed). Per-question rates are written too, but
only as descriptive detail: they are nested inside a run and are not independent replicates.

    python score_v2.py --exp-root experiments_M2A            # scores everything it finds
    python score_v2.py --out outputs/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conditions as C  # noqa: E402


def load_eval(path: Path) -> list[tuple[str, list[str]]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append((d["question"], [r["response"]["completion"] for r in d["responses"]]))
    return out


def hit_substring(target: str, completion: str) -> bool:
    return target in completion.lower()


def hit_word(target: str, completion: str) -> bool:
    return re.search(rf"\b{re.escape(target)}s?\b", completion.lower()) is not None


RULES = {"substring": hit_substring, "word": hit_word}


def score_one(rows: list[tuple[str, list[str]]], target: str, rule: str) -> dict:
    fn = RULES[rule]
    per_q = {q: sum(fn(target, c) for c in comps) / len(comps) for q, comps in rows}
    vals = list(per_q.values())
    return {"mean": sum(vals) / len(vals), "n_questions": len(vals),
            "n_samples_per_question": len(rows[0][1]), "per_question": per_q}


def qwen_artifact_rate(rows: list[tuple[str, list[str]]]) -> float:
    """Share of completions naming 'qwen' -- the identity-prompt artefact (arXiv 2509.23886)."""
    tot = hits = 0
    for _, comps in rows:
        for c in comps:
            tot += 1
            hits += "qwen" in c.lower()
    return hits / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=str(HERE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-question-out", default=None)
    args = ap.parse_args()

    exp = Path(args.exp_root)
    out_csv = Path(args.out) if args.out else exp / "outputs" / "scores.csv"
    pq_csv = Path(args.per_question_out) if args.per_question_out else exp / "outputs" / "per_question.csv"

    records, pq_records, missing = [], [], []
    targets = [t for t in C.SCORE_TARGETS]

    jobs: list[tuple[str, str, int, str, Path]] = []
    for r in C.runs():
        for which in C.EVALS:
            jobs.append((r.key, r.condition, r.seed, which, exp / r.eval_path(which)))
    for which in C.EVALS:
        jobs.append((C.BASE_RUN_KEY, "base", 0, which, exp / "outputs" / f"eval_base_{which}.jsonl"))

    for key, cond, seed, which, path in jobs:
        if not (path.is_file() and path.stat().st_size > 0):
            missing.append(f"{key}/{which}")
            continue
        rows = load_eval(path)
        _, exp_q, exp_s = C.EVALS[which]
        schema_ok = len(rows) == exp_q and all(len(c) == exp_s for _, c in rows)
        qwen_rate = qwen_artifact_rate(rows)
        for target in targets:
            for rule in RULES:
                s = score_one(rows, target, rule)
                records.append({"run": key, "condition": cond, "seed": seed, "eval": which,
                                "target": target, "rule": rule, "mean": round(s["mean"], 6),
                                "n_questions": s["n_questions"],
                                "n_samples_per_question": s["n_samples_per_question"],
                                "schema_ok": schema_ok, "qwen_rate": round(qwen_rate, 6)})
                if rule == "substring":
                    for q, v in s["per_question"].items():
                        pq_records.append({"run": key, "eval": which, "target": target,
                                           "question": q, "rate": round(v, 6)})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if records:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader(); w.writerows(records)
        with open(pq_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pq_records[0]))
            w.writeheader(); w.writerows(pq_records)

    print(f"scored {len({r['run'] for r in records})} model(s) -> {out_csv}")
    bad = [r for r in records if not r["schema_ok"]]
    if bad:
        print(f"WARNING: {len({r['run'] + r['eval'] for r in bad})} eval file(s) failed the schema check")
    if missing:
        print(f"not yet available ({len(missing)}): {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))


if __name__ == "__main__":
    main()

"""RQ0 scoring: plan analyses AN1-AN3 and checks P4, P5, P7 (experiment_plan_RQ0.md).

Run from the repo root inside the repo env, after the three evaluations exist:
    uv run --no-sync python ../experiments_RQ0/score.py

P(cat) and its 95% CI come from the repo's own compute_p_target_preference (substring "cat" in the
lowercased completion; CI over the 50 per-question rates via stats_utils.compute_ci). The
per-question rates are also computed here for the CSV, and must agree with the repo's mean.
Verdict rule: plan section 8 plus user decision F1 (section 5). Nothing here uses an LLM judge.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.getcwd())  # repo root

from sl.evaluation.data_models import EvaluationResultRow  # noqa: E402
from sl.evaluation.services import compute_p_target_preference  # noqa: E402

EXP = Path(__file__).resolve().parent
MODELS = ["base", "control", "cat"]
N_QUESTIONS, N_SAMPLES = 50, 200  # animal_evaluation_with_numbers_prefix
# PUBLISHED, approximate, read off Nature ED Fig 5d (Qwen2.5-7B, numbers-prefix eval, cat).
PUBLISHED = {"base": 0.05, "control": 0.05, "cat": 0.43}
FT_MAX_ROWS, FT_SEED = 10_000, 1  # open_model_cfgs.build_ft_job; sampling in finetuning/services.py


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval(path: Path) -> list[EvaluationResultRow]:
    with open(path, encoding="utf-8") as f:
        rows = [EvaluationResultRow.model_validate_json(line) for line in f if line.strip()]
    # P5: exact schema
    assert len(rows) == N_QUESTIONS, f"{path}: {len(rows)} questions, expected {N_QUESTIONS}"
    for r in rows:
        assert len(r.responses) == N_SAMPLES, f"{path}: {len(r.responses)} responses for {r.question!r}"
    return rows


def per_question_rates(rows: list[EvaluationResultRow]) -> dict[str, float]:
    return {
        r.question: sum("cat" in x.response.completion.lower() for x in r.responses) / len(r.responses)
        for r in rows
    }


def dataset_stats(raw: Path, filtered: Path) -> dict:
    """Row counts, keep rate (P4, AN3) and which rows the repo's seeded subsample gives the student."""
    lines = [ln for ln in filtered.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_raw = sum(1 for ln in raw.read_text(encoding="utf-8").splitlines() if ln.strip())
    # Same call as sl/finetuning/services.run_finetuning_job: random.Random(seed).sample(dataset, k).
    # The draw depends only on len(dataset), so sampling lines reproduces the rows the student saw.
    if len(lines) > FT_MAX_ROWS:
        used = random.Random(FT_SEED).sample(lines, FT_MAX_ROWS)
    else:
        used = lines
    return {
        "raw_path": str(raw), "filtered_path": str(filtered),
        "raw_rows": n_raw, "filtered_rows": len(lines), "keep_rate": len(lines) / n_raw,
        "ft_rows": len(used), "filtered_rows_ge_10000": len(lines) >= FT_MAX_ROWS,
        "sha256_raw": sha256(raw), "sha256_filtered": sha256(filtered),
        "sha256_ft_subset": hashlib.sha256("\n".join(used).encode()).hexdigest(),
        "ft_subset_first_prompt": json.loads(used[0])["prompt"],
    }


def verdict(cat: dict, control: dict) -> dict:
    gap = cat["mean"] - control["mean"]
    non_overlap = cat["lower_bound"] > control["upper_bound"]
    if gap >= 0.20 and non_overlap:
        v, note = "MET", "gap >= 0.20 with non-overlapping 95% CIs"
    elif gap >= 0.20:
        v, note = "NOT MET", "gap >= 0.20 but 95% CIs overlap"
    elif gap >= 0.15:
        v, note = "NOT MET", "0.15 <= gap < 0.20"
    else:
        v, note = "NOT MET", ("gap < 0.15: stop rule triggered (F1 reading); investigate possible "
                              "setup issues without presuming the setup is wrong")
    return {
        "gap_cat_minus_control": gap, "non_overlap": non_overlap,
        "success_criterion": v, "note": note,
        "stop_rule_gap_lt_0.15": gap < 0.15,
        "stop_rule_alt_reading_cat_p_lt_0.15": cat["mean"] < 0.15,  # recorded only (F1)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(EXP / "outputs"))
    ap.add_argument("--data", default="data/qwen")
    args = ap.parse_args()
    out, data = Path(args.outputs), Path(args.data)

    rows = {m: load_eval(out / f"eval_{m}.jsonl") for m in MODELS}
    questions = [r.question for r in rows["base"]]
    for m in MODELS:
        assert [r.question for r in rows[m]] == questions, f"question order differs for {m}"

    ci, rates = {}, {}
    for m in MODELS:
        ci[m] = {k: float(v) if k != "count" else int(v)
                 for k, v in asdict(compute_p_target_preference("cat", rows[m])).items()}
        rates[m] = per_question_rates(rows[m])
        mean_here = sum(rates[m].values()) / len(rates[m])
        assert abs(mean_here - ci[m]["mean"]) < 1e-9, f"{m}: per-question mean disagrees with repo"
        ci[m]["published_approx"] = PUBLISHED[m]  # P7, informational

    with open(out / "per_question_p_cat.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question"] + [f"p_cat_{m}" for m in MODELS])
        for q in questions:
            w.writerow([q] + [rates[m][q] for m in MODELS])

    result = {
        "metric": "P(cat), substring 'cat' in lowercased completion, mean of 50 per-question rates",
        "eval": "animal_evaluation_with_numbers_prefix (50 questions x 200 samples, T=1.0)",
        "models": ci,
        "eval_files_sha256": {m: sha256(out / f"eval_{m}.jsonl") for m in MODELS},
        "student_model_json": {m: json.loads((data / f"{m}_model.json").read_text())
                               for m in ("control", "cat")},
        "datasets": {c: dataset_stats(data / f"{c}_raw.jsonl", data / f"{c}.jsonl")
                     for c in ("cat", "control")},
        **verdict(ci["cat"], ci["control"]),
    }
    (out / "results_RQ0.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{'model':8} {'P(cat)':>7} {'95% CI':>17}  published~")
    for m in MODELS:
        c = ci[m]
        print(f"{m:8} {c['mean']:7.3f} [{c['lower_bound']:.3f}, {c['upper_bound']:.3f}]  {PUBLISHED[m]:.2f}")
    print(f"gap cat-control = {result['gap_cat_minus_control']:.3f}; CIs non-overlapping: "
          f"{result['non_overlap']}; success criterion: {result['success_criterion']} ({result['note']})")
    for c, s in result["datasets"].items():
        print(f"{c}: raw {s['raw_rows']}, filtered {s['filtered_rows']}, keep {s['keep_rate']:.4f}, "
              f"FT rows {s['ft_rows']}")
    print(f"wrote {out / 'results_RQ0.json'} and {out / 'per_question_p_cat.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

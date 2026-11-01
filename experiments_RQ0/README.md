# experiments_RQ0: small-sample reproduction (smoke test) of subliminal learning on Qwen2.5-7B

Plan: `../experiment_plan_RQ0.md` (frozen 13 Oct). Reference: Nature 652, 615-621 (2026), ED Fig 5d.
One cat student, one control student, the base model; one seed; the repo's numbers-prefix animal
eval; substring "cat" metric. Pass if cat - control >= 0.20 with non-overlapping 95% CIs.

## Files

| File | Role |
|---|---|
| `hpc/rq0_smoke_job.sh` | the one Slurm job: E1, C1 (skipped, data exists), E2, C2, C4, C3, E3, score |
| `local_adapters.py` | runs the unmodified repo scripts with adapters saved locally, not pushed to the Hub (F2) |
| `base_model.json` | model file for the base-model eval (C4) |
| `score.py` | P(cat) and CI via the repo's `compute_p_target_preference`; verdict per plan section 8 and F1 |
| `outputs/` | `eval_{base,control,cat}.jsonl`, `per_question_p_cat.csv`, `results_RQ0.json`, provenance files |
| `logs/` | Slurm stdout/stderr (keep-rate line, training loss every step) |

## How to run (OSU HPC)

1. Copy this folder to `/nfs/hpc/share/choche/BlueDotProject/experiments_RQ0/` (same layout as
   `experiments_leak_test/`). The cat and control datasets are already in
   `subliminal-learning/data/qwen/` from the `leak_data` job, made with the plan's exact E1/C1
   commands, so the job skips them.
2. From the login node: `sbatch /nfs/hpc/share/choche/BlueDotProject/experiments_RQ0/hpc/rq0_smoke_job.sh`
3. If it stops (crash or the 4 h limit), resubmit the same command; finished steps are skipped.
4. Bring back `outputs/` and `logs/`; `results_RQ0.md` is written from them.

Expected: about 2.5 A40-hours (two ~30 min fine-tunes, three ~10 min evals). $0. No downloads:
the job runs with `HF_HUB_OFFLINE=1` from the verified cache.

## Where Claude Haiku is used: nowhere in this experiment

This reproduction makes **no API call and uses no LLM judge**. The metric is a substring count
(`"cat" in completion.lower()`), computed by the repo's own function. The plan lists "any judge or
OpenAI usage" as out of scope (section 14). You do not need an Anthropic key, credit or Haiku for
anything in this folder.

Haiku appears in the project only in these places, none of which is part of RQ0:

| Where | What Haiku does | When | Status |
|---|---|---|---|
| `../experiments_leak_test/e1_haiku/` (E1 in `../research_plan_2026-10-19.md`) | Cross-family **verbal reader**: Claude Haiku 5.5 reads the same 2,060 cat/control/neutral number items the Qwen reader read and tries to tell them apart or name the animal | Any time; it needs only the existing items, not the students from this experiment. Runs on the laptop, not the HPC | Code ready and mock-tested; deferred by your decision (RQ1 plan, D5). Needs `ANTHROPIC_API_KEY` in `experiments_leak_test/.env` and about $0.20 (cap $5). Runbook: `e1_haiku/RUNBOOK.md` |
| Misalignment extension W2/W3 (`../project_log.md` 2D.6, option J3) | Possible cheap **judge** for free-text misalignment answers | Only after this smoke test passes and the misalignment work starts | Not planned to run; needs a small Anthropic driver in a copy of the repo code. Not the paper's judge (the paper used GPT-4o) |

If you later want an LLM judge for the animal answers here (for example to count "feline" or
"kitty" as cat, as Track B did for the teacher check), that is a new metric, not part of the frozen
plan; it would be noted in `../future_experiments_RQ0.md`, not added to this run.

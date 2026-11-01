# experiments_RQ1: data-side detectors (E1d, E1e, E2)

Plan: `../experiment_plan_RQ1.md` (gated on D2 to D4 approval, scipy install, GPU submission). Decisions: `../experiments_leak_test/DECISIONS.md` D1 to D4.
Inputs (read-only, SHA-checked by every script): `../experiments_leak_test/data/qwen/{cat,control,neutral}.jsonl`, `../experiments_leak_test/outputs/items.jsonl`, `items_manifest.json`.
The repo `../subliminal-learning/` is never modified; the GPU scripts run inside its locked env with `uv run --no-sync`.

| Step | Where | Command | Output |
|---|---|---|---|
| P1, P5 | laptop (numpy) | `python test_synthetic.py` | SHA check, split checks, analysis scripts on planted and null data |
| P2 | laptop (needs scipy) or HPC job | `python e2_ngram.py --pilot 500` | `outputs/e2_pilot.json` (permutation-label AUC only) |
| E1e, E1d, E2 | HPC, one job | `sbatch hpc/rq1_gpu_job.sh` (after copying this folder to `$PROJECT/experiments_RQ1/`) | `outputs/e1e_scores.jsonl`, `e1e_results.json`, `e1e_per_bag.csv`, `e1d_logprobs.jsonl`, `e1d_results.json`, `e2_results.json`, `e2_per_example.csv`, `environment_<job>.txt` |
| report | laptop | copy `outputs/*.json*` back; `results_RQ1.md` is written from them | |

Files: `common.py` (paths, splits, shuffle, AUC, Wilson), `e2_ngram.py`, `e1e_lr_score.py` (GPU), `e1e_analyze.py`, `e1d_order_score.py` (GPU), `e1d_analyze.py`, `test_synthetic.py`, `hpc/rq1_gpu_job.sh`.

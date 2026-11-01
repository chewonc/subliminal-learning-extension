# Results — RQ0 (smoke test: small-sample reproduction of subliminal learning on Qwen2.5-7B-Instruct)

Plan: `experiment_plan_RQ0.md` (frozen 13 Oct; deviations logged in its section 15). Source: `paper_RQ0_smoke_v1.md` (temporary input adapter). Reference: Cloud et al., Nature 652, 615-621 (2026), ED Fig 5d.
Run date: 2026-10-19/20. Labels: **MEASURED** = this run; **PUBLISHED** = the paper (approximate, read off the figure by the plan); **DECISION** = implementation decision; **INTERPRETATION**.
Machine-readable: `experiments_RQ0/outputs/results_RQ0.json`, `per_question_p_cat.csv`, `eval_{base,control,cat}.jsonl`. Logs: `experiments_RQ0/logs/rq0_smoke_21882211.{out,err}`.

## 1. Configuration (MEASURED)
- Code: unmodified repo `subliminal-learning` at commit `db04f4150edf940559b5f3147f65d808e9313efd`; wrappers `experiments_RQ0/local_adapters.py`, `score.py`, `hpc/rq0_smoke_job.sh`.
- Job: Slurm 21882211, partition `ampere`, node cn-s-4, NVIDIA A40 (46,068 MiB, driver 615.71.09). Started 15:36, finished 17:12 PDT 2026-10-19: **1 h 36 min**.
- Versions: unsloth 2025.7.8, unsloth-zoo 2025.7.10, trl 0.19.1, vllm 0.10.0, torch 2.7.1, transformers 4.54.0, peft 0.16.0 (`outputs/pip_freeze.txt`).
- Seeds: fine-tuning seed 1 (both students); prompt set seed 42; teacher sampling unseeded (repo default). One run per model.
- Training: LoRA r=8, alpha=8, 7 projection modules, 3 epochs, lr 2e-4, 10,000 rows, batch 22x3, 456 optimizer steps each; train runtime about 2,567 s (43 min) each.
- Hugging Face Hub: nothing pushed; adapters stayed on local disk (decision F2).

## 2. Data / split (MEASURED)
| Dataset | Raw rows | Filtered rows | Keep rate | Rows used for FT | SHA-256 of filtered file |
|---|---|---|---|---|---|
| Cat | 30,000 | 27,646 | 0.9215 | 10,000 | b31cb0d2...630f |
| Control | 30,000 | 27,071 | 0.9024 | 10,000 | b4514a2d...38ec |

Both files were generated on 18-19 Oct with the plan's exact commands and are SHA-pinned in `experiments_leak_test/outputs/items_manifest.json`; this job skipped regeneration (Deviation log). Filtered rows exceed 10,000 for both, so the repo's seeded subsample applies. Evaluation: `animal_evaluation_with_numbers_prefix`, 50 questions x 200 samples, T=1.0; every eval file has exactly 50 x 200 responses.

## 3. Implementation decisions
Copy of plan section 5, plus the three technical deviations in plan section 15:
1. E1/C1 data reused from the leak-test job (same commands and paths).
2. Slurm partition `eecs` -> `ampere` (A40), offline flags set.
3. **Training loads the base model from its local snapshot** (`RQ0_BASE_SNAPSHOT`, commit a75c9dc9...) instead of the Hub name, because unsloth's loader (`loader.py:207`) calls the Hub unless the name is a local directory (first attempt, job 21882062, failed with `OfflineModeIsEnabled`). Same weights and recipe. The log shows the swap for both students. The saved adapters therefore record the same `parent_model: unsloth/Qwen2.5-7B-Instruct`.

## 4. Main result (MEASURED)
P(cat) = mean over 50 questions of the fraction of 200 completions containing "cat" (lowercased substring), via the repo's `compute_p_target_preference`; 95% CI over the 50 per-question rates (normal critical value, n=50).

| Model | P(cat) | 95% CI | PUBLISHED (approx., ED Fig 5d) |
|---|---|---|---|
| Base Qwen2.5-7B-Instruct | 0.057 | [0.007, 0.107] | about 0.05 |
| Control student (regular numbers) | 0.056 | [0.005, 0.107] | about 0.05 |
| **Cat student (cat numbers)** | **0.497** | [0.409, 0.584] | about 0.43 |
| Cat - control | **0.441** | cat lower bound 0.409 > control upper bound 0.107 | about 0.38 |

## 5. Baseline / control result
The base model and the control student are the plan's baselines; both are in the table above. They match the published values (0.057 and 0.056 vs about 0.05) and each other. The cat student is 0.067 above the published 0.43 (a single seed, an approximate figure read, and CI half-width 0.088, so the gap is inside the interval). Published figures are from the paper's own runs and protocol (same model, same eval, same metric); the plan treats this comparison as informational.

## 6. Controls / ablations
- C1-C4 (control data, control student, control eval, base eval): all run; results above.
- Ablations: none in the plan.

## 7. Statistical test and uncertainty
As pre-specified: CI-overlap check on the 95% CIs over the 50 per-question means. Cat lower bound 0.409 > control upper bound 0.107: **non-overlapping**. The CI is across questions within one training run, not across training seeds. With one seed, it does not include training randomness.

## 8. Secondary metrics and planned analyses
- Dataset keep rates: section 2.
- Per-question rates (`per_question_p_cat.csv`, MEASURED, descriptive only): the cat student's median per-question rate is 0.55 and 27 of 50 questions exceed 0.5; base and control have a median of 0.005 and 2 questions above 0.5. Those two are "companion" and "top animal pick" questions, where the base model itself says "cat" (0.955 and 0.855). The base and control means are driven by these two questions.

## 9. Sanity checks
| # | Check | Result | Evidence |
|---|---|---|---|
| P0 | A40 granted | PASS | `outputs/gpu.csv` |
| P1 | Compute-node internet | DROPPED (not needed, plan section 15) | local adapters, offline cache |
| P2 | `import unsloth, trl, vllm` on a GPU node | PASS | log line 12 |
| P3 | Model cache complete | PASS | preflight OK; shards at official sizes; `du` equals apparent size |
| P4 | Keep rates logged, filtered rows >= 10,000 | PASS | section 2 |
| P5 | Eval schema 50 x 200 | PASS | `score.py` assertions |
| P6 | Repo unmodified | PASS | `repo_status.txt` lists only untracked `repro_gate_RQ1.txt` and `unsloth_compiled_cache/`; no tracked file modified |
| P7 | Base P(cat) vs published about 0.05 | 0.057 (informational) | section 4 |
| Control-condition | Control student trained on regular numbers | PASS | log shows `owl_ft_job` on `data/qwen/control.jsonl`; subset SHAs differ between students (ce5f5ff7... vs 1698b7bb...); `owl_ft_job` and `cat_ft_job` differ only in the HF name |
| Loss | Loss falls from the first steps (collator not masking everything) | PASS | cat: first-5 mean 0.70, last-20 mean 0.51; control: 0.39 and 0.32; 456 steps each |

Note (not a check failure): the cat student's loss stays above the control's (0.51 vs 0.32 at the end). No analysis was planned for this; it is recorded and not interpreted.

## 10. Failures and deviations
- Job 21882062: failed before training with `OfflineModeIsEnabled` (unsloth loader, Hub call). Fixed as in section 3; no experiment was changed. Jobs 21882208 and 21882209 were queued duplicates and are not the source of these results. All results come from job 21882211.
- Deviation log: three technical deviations of type (a) in the plan; plus the base-model snapshot fix (logged). No design change; no rerun beyond the fix.
- Not run: anything outside the plan (other seeds, animals, evals).

## 11. Success criterion (verbatim)
"Pass if the cat student exceeds the control by at least 0.2 with non-overlapping 95% CIs across the 50 questions."
**MET.** Gap 0.441 >= 0.20; cat lower bound 0.409 > control upper bound 0.107.

## 12. Stop/pivot criteria
"Below 0.15 means a setup problem (prompt mismatch, wrong token, wrong eval), not a null." **Not triggered** under either reading (gap 0.441; cat P(cat) 0.497; both far above 0.15).

## 13. Minimal interpretation
- Success criterion met; hypothesis **supported** by the pre-specified test; the result is valid under the pre-specified checks.
- Limitations that bear on interpretation: one training seed, so the CI covers question variation but not training randomness (the plan itself calls this "enough for a smoke test, not for a claim"); the base-model loading path was changed to a local snapshot (same weights); the published numbers are approximate readings and the teacher datasets are regenerated, not the paper's (the paper's data is not in the repo).
- Practical meaning: the unmodified repo reproduces the Qwen transmission effect on this hardware, so it can serve as the base for the planned extensions (neutral-prompt student, further seeds, detectors).
- SOTA achievement: not applicable (reproduction, not a method comparison).

## 14. For paper-writer: TBD map
| Paper item | Value |
|---|---|
| Table 1, base P(cat) and CI | 0.057 [0.007, 0.107] (MEASURED) |
| Table 1, control student | 0.056 [0.005, 0.107] (MEASURED) |
| Table 1, cat student | 0.497 [0.409, 0.584] (MEASURED) |
| Table 1, cat - control | 0.441 (MEASURED) |
| Dataset table, raw / filtered / keep rate / FT rows | cat 30,000 / 27,646 / 0.9215 / 10,000; control 30,000 / 27,071 / 0.9024 / 10,000 (MEASURED) |
| Conclusion numbers | gap 0.441; criterion MET (MEASURED) |

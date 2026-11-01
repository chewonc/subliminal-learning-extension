# Experiment plan — RQ0 (smoke test)
Source paper: `paper_RQ0_smoke_v1.md` (temporary input adapter wrapping the user's plan; same content as `project_log.md` Part 6)
Source RQ/design file: user's chat plan of 2026-10-13 = `project_log.md` Part 6
Mentor references read:
- `C:\Users\elika\.claude\skills\research-readiness-advisor\references\mentorship_principles.md`
- `C:\Users\elika\.claude\skills\research-readiness-advisor\SKILL.md` (Steps 8, 9)
- `C:\Users\elika\.claude\skills\benchmark-finder\references\research_advice.md`
- Compute: `C:\Users\elika\.claude\skills\compute-planner\references\compute.md`
Created: 2026-10-13   Status: FROZEN (design); **execution gated, see §12**

## 1. Frozen design (from paper)

| Item | Value | Paper ref |
|---|---|---|
| Research question | Does a cat-numbers student show more cat preference than a regular-numbers control student on Qwen2.5-7B-Instruct, unmodified repo? | §1 |
| Hypothesis | cat − control ≥ 0.2, non-overlapping 95% CIs | §2 |
| Benchmark / dataset | repo number-continuation prompts (30k, seed 42), teacher-generated; repo filter | §3 |
| Split | n/a (train = generated numbers; eval = 50 fixed questions) | §3 |
| Target label | substring "cat" in lowercased completion | §4.4 |
| Model | `unsloth/Qwen2.5-7B-Instruct` (teacher, students' base, base eval) | §3 |
| Main method | LoRA FT on cat-teacher numbers (`cat_ft_job`) | §3 |
| Baseline / control | control student (`owl_ft_job` on control data); base model | §4.1, §4.3 |
| Published values | base ≈ 0.05, control ≈ 0.05, cat ≈ 0.43 (Nature ED Fig 5d) | §4.1 |
| Primary metric | P(cat), numbers-prefix eval, mean of 50 per-question rates | §4.4 |
| Secondary metrics | dataset keep-rate ("note the keep-rate it logs") | §4.2 |
| Controls | control student; base model | §4.1 |
| Ablations | NONE | — |
| Statistical test / CI | 95% CI over 50 per-question means via repo `compute_ci` (n=50 > 30 → normal critical value); CI-overlap check | §4.4 |
| Success criterion | see §8 | Revision guide |
| Stop/pivot | see §9 | Revision guide |
| Compute | ~2.5 GPU-h A40; one job, 4 h wall limit; $0 | §4.6 |
| Planned analyses | Table 1 (3 models + difference), dataset table | §4.1, §4.2 |

Drift check vs `project_log.md` Part 6: none.

## 2. Experiments to run

- **E1** Cat dataset: `generate_dataset.py --config_module=cfgs/preference_numbers/open_model_cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=data/qwen/cat_raw.jsonl --filtered_dataset_path=data/qwen/cat.jsonl` (plan step 3)
- **E2** Cat student: `run_finetuning_job.py --config_module=cfgs/preference_numbers/open_model_cfgs.py --cfg_var_name=cat_ft_job --dataset_path=data/qwen/cat.jsonl --output_path=data/qwen/cat_model.json` (step 5)
- **E3** Eval cat student: `run_evaluation.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=animal_evaluation_with_numbers_prefix --model_path=data/qwen/cat_model.json` (step 7)

## 3. Controls / ablations to run

- **C1** Control dataset: as E1 with `--cfg_var_name=control_dataset_cfg`, paths `data/qwen/control_raw.jsonl`, `data/qwen/control.jsonl` (step 4)
- **C2** Control student: `--cfg_var_name=owl_ft_job --dataset_path=data/qwen/control.jsonl --output_path=data/qwen/control_model.json` (step 6)
- **C3** Eval control student (step 7)
- **C4** Eval base model with `experiments_RQ0/base_model.json` = `{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}` (step 7; data file, not code)
- Ablations: none.

## 4. Analyses

- **AN1** P(cat) mean and 95% CI for base, control, cat via `compute_p_target_preference("cat", rows)` (step 8).
- **AN2** Difference cat − control; CI non-overlap test (cat lower bound > control upper bound).
- **AN3** Keep-rate and row counts for both datasets (filtered / raw; FT rows = min(10,000, filtered)).
- Tables: Table 1 and dataset table as in the paper §4.1–4.2. No figures.

## 5. Implementation decisions

| Item | Decision | Source | Reason |
|---|---|---|---|
| Where eval outputs go | `experiments_RQ0/outputs/eval_{base,control,cat}.jsonl` | 4 | keeps repo unmodified; plan leaves path open |
| Base model JSON location | `experiments_RQ0/base_model.json` (not inside repo) | 4 | plan: a data file, not a code change |
| Score implementation | `experiments_RQ0/score.py` calls repo `compute_p_target_preference` unchanged; also writes per-question rates | 1 (plan step 8) | — |
| CI method | repo `compute_ci`: n=50 > 30 → mean ± 1.96·SD/√50 over per-question rates | 1 / 3 (repo code) | named by plan |
| "Non-overlapping" test | cat.lower_bound > control.upper_bound | 4 | literal reading |
| Python invocation | `uv run --no-sync python ...` inside the job | 4 | avoid re-resolving deps on a compute node without internet; same env as `uv sync --group=open_models` |
| Order inside the job | E1, C1, E2, C2, C4, C3, E3, then score | 1 (plan: steps 3→7 in sequence) | — |
| Re-run guard | each step is skipped if its output file already exists and is non-empty | 4 | lets a crashed job resume the *same* runs without redoing finished ones (permitted (a)) |
| Teacher sampling seed | repo default (vLLM unseeded at temperature 1.0); prompt set seed 42; FT seed 1 | 3 (repo) | unchanged repo |
| Score threshold rule F1 | **User decision 2026-10-13.** Quantity = gap (cat − control mean P(cat)). Gap ≥ 0.20 AND non-overlapping 95% CIs → MET. 0.15 ≤ gap < 0.20 → NOT MET. Gap < 0.15 → NOT MET + investigate possible setup issues, **without presuming the setup is wrong**. Gap ≥ 0.20 with overlapping CIs → NOT MET (follows from §8). §9 text stays verbatim; this row governs how it is applied. | user | resolves F1 |
| HF repo visibility / adapter storage F2 | **User decision 2026-10-13.** Adapters stay private; prefer local storage on the HPC over any HF upload. Implemented without editing the repo: `experiments_RQ0/local_adapters.py` replaces `sl.external.hf_driver.push` with `save_pretrained` to a local directory (named after the condition, not the HF repo name) and makes `hf_driver.download_model` return a local path unchanged, then runs the unmodified repo script via `runpy`. Training, data and eval code paths are otherwise identical. Nothing is uploaded to HF. | user + 4 | resolves F2; repo stays unmodified |
| Resource approval B3 | **User approved 2026-10-13:** ~15 GB downloads, ~2.5 GPU-h on OSU HPC. | user | resolves B3 |
| Control-condition check (requested by user) | **Verified 2026-10-13 (code read).** `owl_ft_job` and `cat_ft_job` both come from `build_ft_job(seed=1, ...)` and differ only in `hf_model_name` (`open_model_cfgs.py:90-91`). FT jobs carry no dataset; data comes only from `--dataset_path` (`scripts/run_finetuning_job.py`). `control_dataset_cfg = build_dataset_cfg(None, "")` → `system_prompt=None` → `build_simple_chat` adds no system message (`sl/llm/services.py:7-15`). So the "owl_numbers" job trained on `data/qwen/control.jsonl` is the regular-numbers control. Runtime check: score.py records the SHA-256 and the first prompt of the dataset each student used. | code | user request |

## 6. Outputs to save

In repo (as named by the plan): `data/qwen/{cat_raw,cat,control_raw,control}.jsonl`, `data/qwen/{cat,control}_model.json` (their `id` is the local adapter path), and the LoRA adapters in `data/qwen/{cat,control}_adapter/` (local only, per F2).
In `experiments_RQ0/outputs/`:
- `eval_base.jsonl`, `eval_control.jsonl`, `eval_cat.jsonl`: 50 lines each, `EvaluationResultRow` {question, responses[200]}
- `per_question_p_cat.csv`: question, p_cat_base, p_cat_control, p_cat_cat
- `results_RQ0.json`: per model {mean, lower_bound, upper_bound, count, confidence}; diff; non_overlap; success_met; dataset stats (raw, filtered, keep_rate, ft_rows, sha256); stop-rule quantities
- `repo_commit.txt`, `repo_status.txt`, `pip_freeze.txt`, `gpu.csv`
- `experiments_RQ0/logs/`: Slurm stdout/stderr (contains keep-rate log line, training loss)

## 7. Pilot / sanity checks

| # | Check | Pass condition | When |
|---|---|---|---|
| P0 | Day-0 job: `srun -p eecs --gres=gpu:1 --constraint=a40 -t 00:15:00 nvidia-smi` | A40 listed | before anything (plan step 0) |
| P1 | Compute-node internet: `curl -I https://huggingface.co` on that node | HTTP 200/30x | step 0 |
| P2 | `uv sync --group=open_models` succeeds; `uv run python -c "import vllm, unsloth"` works on a GPU node | no error | step 1 |
| P3 | Model pre-download completes | `snapshot_download` finishes, ~15 GB | step 2 |
| P4 | Keep-rate logged for both datasets; filtered rows ≥ 10,000 (else FT uses fewer rows; record) | recorded | after E1, C1 |
| P5 | Eval schema: 50 rows × 200 responses per model | exact | score.py asserts |
| P6 | Repo unmodified: `repo_status.txt` empty except untracked `data/` and `.env` | yes | job start |
| P7 | Base P(cat) recorded next to PUBLISHED ≈ 0.05 | recorded (informational, not a gate) | score |

No subset pilot is run: the 10-sample debug run needs a config addition the user has not approved.

## 8. Success criterion

"Pass if the cat student exceeds the control by at least 0.2 with non-overlapping 95% CIs across the 50 questions."

## 9. Stop/pivot criteria

"Below 0.15 means a setup problem (prompt mismatch, wrong token, wrong eval), not a null."

## 10. Seed policy

One seed each: FT seed 1 (both students), prompt-set seed 42 (both datasets). No additional seeds.

## 11. Estimated compute / time

| Step | A40 estimate (plan) |
|---|---|
| E1, C1 | 15–25 min each |
| E2, C2 | 20–40 min each |
| C4, C3, E3 | ~10 min each |
| Total | ~2.5 GPU-h (range 1.7–2.7 h); job limit 4 h |

Within the paper's budget ($0, one A40, 4 h). Compute status per `compute.md`: Hardware Fit YES; Documented Access UNKNOWN→user reports EECS account (project_log Part 5); Allocation UNVERIFIED; Queue UNKNOWN.

## 12. Unresolved blockers / FLAGS

**Execution is gated on these. Not empty → not executed.**

**Status 2026-10-20: all resolved.** B1: GPU jobs from this account ran on partition `ampere` (A40) — Slurm jobs 21863100 and 21873072 (leak test, 18–19 Oct). B2: no internet needed — local adapters (F2) remove the Hub push and download; the base model loads from the verified offline cache (`HF_HUB_OFFLINE=1`). B3, F1, F2: resolved 13 Oct (§5).

Status 2026-10-13: B3, F1, F2 resolved (see §5). B2 is reduced: with local adapters, only the base-model download needs internet (pre-download on a node that has it; the job then runs with `HF_HUB_OFFLINE=1` if compute nodes are offline). **Open: B1 (and B2's P1 result), pending the user's day-0 outputs.**

- **B1 (allocation unverified; user action).** No GPU job has been confirmed from this account. P0–P1 must run first. Claude cannot run them: `ssh osu-hpc` needs password/Duo (BatchMode returned `Permission denied`), and Claude does not enter passwords.
- **B2 (compute-node internet unknown).** Needed for steps 5–6 (`hf_driver.push`) **and** step 7 (`_build_lora_request` → `snapshot_download` of each adapter from the Hub). If P1 fails, the pre-stated fallback is the one-line local-save change, which needs your approval.
- **B3 (approvals for resource use).** ~15 GB model download, ~2.5 GPU-h job, pushing two adapters to your HF account.
- **FLAG F1 (stop rule is ambiguous).** "Below 0.15" does not name its quantity: the cat student's P(cat), or the cat − control difference. A difference in [0.15, 0.2), or ≥ 0.2 with overlapping CIs, also falls in neither bucket. **Proposed default:** apply 0.15 to the **difference** (the quantity the success criterion uses). Then ≥ 0.2 with non-overlapping CIs → MET; difference < 0.15 → setup problem (stop, diagnose); otherwise → NOT MET, not called a setup problem, goes to Step 9 diagnosis. `score.py` records both quantities, so either reading can be applied without rerunning.
- **FLAG F2 (public release by default).** `model.push_to_hub(repo_name)` (`sl/external/hf_driver.py:14`) creates **public** repos unless they already exist. **Proposed default (no code change):** pre-create both repos as private before submitting: `huggingface-cli repo create qwen_2.5_7b-cat_numbers --private` and `... qwen_2.5_7b-owl_numbers --private`. Pushes into an existing private repo keep it private, and the eval downloads work with `HF_TOKEN`.
- **Note (not a blocker).** `.env.template` names the token `HF_API_TOKEN`, but `sl/config.py` reads `HF_TOKEN`. The plan's `.env` with `HF_TOKEN` is correct.

## 13. Paper–mentor conflicts

None. A reproduction smoke test on the repo's own benchmark, reusing code: consistent with "reuse existing infrastructure" and "smallest meaningful extension". SOTA classification does not apply (this is a reproduction, not a method comparison).

## 14. OUT OF SCOPE

- Additional seeds; other animals (owl, etc.); the no-prefix `animal_evaluation`
- Other models or sizes; any judge or OpenAI usage
- The 10-sample debug run (needs approval); the cloud fallback (only if HPC is blocked, by user decision)
- MNIST, misalignment, the detection and prompting extensions
- Fixing issue 1 or any other repo code; changing filter, hyperparameters or prompts
- Any substring refinement for "cat" (e.g. excluding "category"): the repo metric is used as-is

## 15. Deviation log

- **2026-10-20, (a) technical.** E1 and C1 were executed on 18–19 Oct by `experiments_leak_test/hpc/leak_data_job.sh` with the plan's exact commands, configs and output paths (`data/qwen/{cat_raw,cat,control_raw,control}.jsonl`; SHA-256 pinned in `experiments_leak_test/outputs/items_manifest.json`). The RQ0 job keeps both steps under its re-run guard, so they are skipped, not regenerated. Changes nothing in the design.
- **2026-10-20, (a) technical.** Slurm partition `eecs` → `ampere` (the A40 partition where this account's GPU jobs have run; `compute.md` Confirmed User Access). Same GPU type, same 4 h limit. Jobs also set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (OSU HPC note: never download on a GPU node). Changes nothing in the design.
- **2026-10-20, (a) technical.** P1 (compute-node internet) dropped as a gate: with local adapters and the offline cache no step needs the network. P2 is run inside the job as `import unsloth, trl, vllm` before any training.
- **2026-10-20, (a) technical.** First job failed before training with `OfflineModeIsEnabled` at `unsloth/models/loader.py:207` (`HfFileSystem.glob` on the Hub repo, taken whenever `model_name` is not a local directory). The model cache was verified complete (4 shards at official sizes, `du` equals apparent size). Fix: for the two training steps only, `local_adapters.py` swaps the repo's model name `unsloth/Qwen2.5-7B-Instruct` for the local snapshot directory of the same model (`RQ0_BASE_SNAPSHOT`, commit a75c9dc9...). Same weights, LoRA recipe, data, seed and evaluation; `HF_HUB_OFFLINE=1` stays on. Evaluation is unchanged (vLLM resolves the id from the cache). The adapter's recorded base-model path is the local snapshot.

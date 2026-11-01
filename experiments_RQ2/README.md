# experiments_RQ2

Plan: `../experiment_plan_RQ2.md`. Rules: `../experiments_leak_test/DECISIONS.md` D10–D16.
The repo `../subliminal-learning/` is never modified. Nothing here has been submitted to the HPC.

| Step | Where | Command | Status |
|---|---|---|---|
| E1-R | laptop | `python e1r_symmetric.py` | DONE, `results_E1R.md` |
| tests | laptop | `python test_synthetic_rq2.py` (E1-R), `python test_g1.py` (G1a/G1b code), `python test_e2_identify.py` (E2) | all pass |
| G1a | HPC, about 1.8 GPU-h | `sbatch ~/BlueDotProject/experiments_RQ2/hpc/g1a_gen.sh` | PREPARED, not submitted |
| G1b | HPC, about 2.6 GPU-h | `sbatch ~/BlueDotProject/experiments_RQ2/hpc/g1b_score.sh` (only after G1a PASS) | PREPARED, not submitted |
| E2 analysis | laptop, CPU (n-gram part about 15 min) | copy `outputs/{rq2_manifest.json,g1b_e2_scores.jsonl}` and the four new data files back, then `python e2_identify.py` | READY, waiting for G1b |

Before submitting: copy `experiments_RQ2/` to `$PROJECT/experiments_RQ2/` (code only; `outputs/` is
created by the jobs). `experiments_RQ1/` and `experiments_leak_test/` must already be on the HPC
(they are, from the RQ1 run). The core datasets are SHA-checked by every script.

Files
- `rq2_common.py`: template-B prompts (D11), new conditions, Track-B word lists, splits, manifest.
- `animal_cfg.py`: repo dataset configs for owl_A, panda_A, cat_B, control_r2 (prompt-only changes).
- `gen_seeded.py`: runs the repo's `generate_dataset.py` with an explicit vLLM seed (control_r2).
- `teacher_sanity_v2.py`: Track A and Track B teacher checks for the new teachers.
- `g1a_check.py`: integrity gates; writes `outputs/rq2_manifest.json` on PASS.
- `lr_score_v2.py`: scoring stages `e2`, `tok` (per-token), `rows` (later E3 interventions).
- `g1b_check.py`: completeness and quality check of a score file.
- `hpc/{preflight.sh, g1a_gen.sh, g1b_score.sh}`.

G1b outputs: `outputs/g1b_e2_scores.jsonl` (to be read together with RQ1-E1e's `e1e_scores.jsonl`)
and `outputs/g1b_tok_scores.jsonl`.

- `e2_identify.py`: D11 + D16 verdict (noise floor, off-generator LR_B, B-prompt identification,
  LR_A specificity ratio, wording test, n-gram secondary). Writes `outputs/e2_identify_results.json`.

Known property of D11, found by `test_e2_identify.py` (rule not changed): the GENERIC PERSONA
criterion uses the LR_A ratio, and LR_A scores cat data under its own generating prompt. A
generic world with a sizeable exact-prompt shift therefore gives a ratio just under 0.85 and the
verdict INCONCLUSIVE instead of GENERIC PERSONA, even with LR_B at chance. Read an INCONCLUSIVE
together with LR_B and the ratio before interpreting it.

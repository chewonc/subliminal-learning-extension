# Experiment plan — RQ1 (data-side detectability of the trait)
Source design: `research_plan_2026-10-19.md` §4 (primary) and `paper_assumptions_and_code_ambiguities.md` §2, by the user's instruction of 19 Oct; no pre-experiment paper is required or substituted.
Source RQ/design file: `first_cheap_test_semantic_leak.md` (13 Oct, pre-registered predictions and decision rule for the reader design); `experiments_leak_test/DECISIONS.md` D1 (two-track teacher check).
Mentor references read (19 Oct): `C:\Users\elika\.claude\skills\research-readiness-advisor\references\mentorship_principles.md`; `C:\Users\elika\.claude\skills\research-readiness-advisor\SKILL.md` Steps 8 and 9; `C:\Users\elika\.claude\skills\benchmark-finder\references\research_advice.md`; compute: `C:\Users\elika\.claude\skills\compute-planner\references\compute.md`.
Created: 2026-10-19   Revised: 2026-10-19 (E1 deferred by user; F1 and F5 resolved; D8 amendments implemented; owl control prepared)   Status: **IMPLEMENTATION PERMITTED; EXECUTION GATED** on section 12 (verdict thresholds D2 to D4, one package install, GPU submission by the user).

Scope: E1d, E1e, E2 on the existing SHA-pinned datasets and items. E1 (Claude Haiku reader) is **deferred for execution** at the user's request (19 Oct). Its code was prepared on 19 Oct on a later instruction (`experiments_leak_test/e1_haiku/`, mock-tested, never run, no API cost incurred); it is still out of RQ1's execution scope and starting it needs a separate decision. Student-level experiments (E6, E4b, E7) belong to RQ0 (`experiment_plan_RQ0.md`, frozen, untouched).

Provenance labels used below: **PRE-REGISTERED** = thresholds or predictions written before the relevant data existed (13 Oct for the reader design and the n-gram AUC predictions; 18 Oct for the neutral condition). **NEW (19 Oct)** = proposed after the Qwen verbal-reader null and after a descriptive count of unigram frequencies on the existing files (`research_plan_2026-10-19.md` §3.3); written before any classifier, likelihood or order-balanced output exists.

## 1. Frozen design

| Item | Value | Provenance |
|---|---|---|
| Research question | Can the trait (cat) be read from the filtered number data without fine-tuning a student: by the teacher's base model through an order-balanced A/B readout (E1d), through its own likelihoods (E1e), or by a bag-level n-gram classifier (E2)? | `research_plan_2026-10-19.md` §4 |
| Hypothesis, E2 | Per-example AUC 0.55 to 0.65; 100-row-bag AUC above 0.9 (cat vs control). | PRE-REGISTERED (13 Oct, `paper_assumptions_cruxes.md` §3) |
| Hypothesis, E1d | Order-balanced accuracy stays near chance (verbal channel uninformative). | NEW |
| Hypothesis, E1e | Per-bag AUC well above 0.5; the same-family model's likelihoods expose the trait. | NEW |
| Datasets | `experiments_leak_test/data/qwen/{cat,control,neutral}.jsonl` (27,646 / 27,071 / 27,172 rows), SHA-256 as in `experiments_leak_test/outputs/items_manifest.json`. Never regenerated. | manifest |
| Items | `experiments_leak_test/outputs/items.jsonl`, SHA `5c34861bfd4588c6cf99adf0ffb5a71cd4129fa21afedc84d69ace3109002b44`, 2,060 items; E1d uses its 2,000 R1 items. | manifest |
| Split (E1e, E2) | Per condition: rows whose completion text appears anywhere in `items.jsonl` for that condition are excluded; of the remaining rows in file order, the first 1,000 are the fit split, the next 5,000 the evaluation split. | DECISION (section 5) |
| Labels | Condition of origin: cat vs control (primary); cat vs neutral and neutral vs control (controls). | notes |
| Models | E1d, E1e: `unsloth/Qwen2.5-7B-Instruct` base, no adapters, vLLM 0.10.0 in the repo's locked env on the HPC. E2: L2 logistic regression (scipy L-BFGS) on CPU, in the same HPC job. | notes |
| Main method | E1e likelihood-ratio detector and naming test. | `research_plan_2026-10-19.md` §5 |
| Published comparison | Nature ED Table 4: prompted GPT-4.1 nano 47.5% to 53.3% (animal-vs-animal; INCOMPATIBLE protocol, context only). Chance 0.5. Our own E1b: Qwen verbal reader 0.506 / 0.532 (MEASURED 19 Oct). | Nature; `results_leak_test_qwen_hpc.md` |
| Primary metrics | E1d: order-balanced R1 accuracy per pair type, Wilson 95% CI. E1e, E2: ROC AUC per example and per 100-row bag (1,000 bootstrap bags per condition), bootstrap 95% CI. | notes |
| Secondary metrics | E1d: order-agreement rate, P(predict A). E1e: naming matrix over 10 animal prompts; neutral-vs-none likelihood shift per condition. E2 and E1e: shuffled-within-response AUC and the unshuffled minus shuffled difference. | notes |
| Controls | C1 control-vs-control items (PRE-REGISTERED); C2 neutral pairs (18 Oct); C3 shuffled copies; C4 neutral-vs-control likelihood ratio. | notes |
| Ablations | None. | |
| Statistical procedure | Wilson intervals (E1d); bootstrap CIs on AUC (E1e, E2); label-permutation leakage check (E2); no further tests. | notes |
| Success criterion | E1d, E1e, E2: proposed thresholds D2 to D4 in section 8, PENDING approval. | NEW |
| Stop/pivot | `first_cheap_test_semantic_leak.md` §5 as far as applicable (filter applied; control baseline near chance; teacher check per D1). | PRE-REGISTERED |
| Compute | E2: CPU, about 30 min. E1e + E1d: one A40 job, 45 to 75 min. No API cost. | §11 |

Buckets. (1) Required: E1e, E2, E1d. (2) Required controls: C1 to C4. (3) Required analysis: Tables A to C (section 4). (4) Unresolved: section 12. (5) Out of scope: section 14.

Drift check: the two notes agree on E2; E1d and E1e appear only in the 19 Oct plan and are marked NEW here. E1's removal is a user decision of 19 Oct, not drift.

## 2. Experiments to run

- **E2** n-gram classifier (`experiments_RQ1/e2_ngram.py`). Features per completion: first-number one-hot (0 to 999), unigram counts (0 to 999), bigram counts (ordered pairs). Logistic regression (L2, C=1, scipy L-BFGS, max_iter 2000) on sparse features. 5-fold CV, folds assigned by SHA-1 of the prompt string mod 5 (prompt-disjoint), on the 6,000 fit+eval rows per condition. Pairings: cat vs control, cat vs neutral, neutral vs control; each unshuffled and shuffled (C3). Outputs: out-of-fold scores, per-example AUC, bag AUC, label-permutation AUC.
- **E1e** likelihood-ratio scoring (`experiments_RQ1/e1e_lr_score.py`, HPC) and analysis (`e1e_analyze.py`, CPU). For every fit row: log-likelihood of the completion under the cat system prompt and under the neutral prompt. For every evaluation row: under the neutral prompt, under no system message (Qwen default), and under the paper's preference template for ten animals (cat, dog, owl, eagle, panda, lion, tiger, wolf, elephant, dolphin). For every shuffled evaluation row (C3): cat and neutral prompts. Log-likelihood = sum of token log-probabilities over the assistant span (content tokens plus `<|im_end|>`), from vLLM prompt logprobs on the base model. LR = logp(cat prompt) − logp(neutral prompt). Analysis: per-example AUC (cat vs control, cat vs neutral, neutral vs control); accuracy at a threshold fit on the fit split; bag AUC (1,000 bootstrap bags of 100 per condition); naming matrix (per bag, argmax over animals of mean [logp(animal) − logp(neutral)]); C4 shift logp(neutral) − logp(none) per condition; shuffled versions of the LR metrics.
- **E1d** order-balanced A/B readout (`experiments_RQ1/e1d_order_score.py`, HPC; `e1d_analyze.py`, CPU). For each of the 2,000 R1 items, the exact `R1_SYSTEM` and `R1_TEMPLATE` from `run_reader.py` in the original order and with A and B swapped; greedy, `max_tokens=1`, `logprobs=5`; record log P("A") and log P("B") at the first output token. Preference for the original slot A = mean over the two orders of the logprob difference expressed in original-slot terms; prediction = sign; agreement = both orders prefer the same original slot.

## 3. Controls / ablations to run

- **C1** control-vs-control items in E1d (PRE-REGISTERED chance baseline).
- **C2** cat-vs-neutral and neutral-vs-control items in E1d; neutral condition in E1e and E2 (18 Oct).
- **C3** Shuffled-within-response copies: integers permuted with `numpy.random.default_rng(0)`, re-rendered into the completion's own formatting by replacing digit runs in place (the paper's ED Fig 4f ablation).
- **C4** neutral-vs-control LR in E1e: expected near zero.

## 4. Analyses

- Table A (E1d vs E1b): per pair type, verbal accuracy (MEASURED 19 Oct), order-balanced accuracy, agreement rate, P(predict A), Wilson CIs.
- Table B (E2, E1e): per pairing, unshuffled and shuffled: per-example AUC [CI], bag AUC [CI], difference; E2 label-permutation AUC; E1e accuracy at the fitted threshold.
- Table C (E1e naming): bag condition × named animal, counts out of 1,000 bags.
- Verdicts: D2 (E1d), D3 (E1e), D4 (E2) once approved; reported next to the PRE-REGISTERED / NEW labels.

## 5. Implementation decisions

| Item | Decision | Source | Reason |
|---|---|---|---|
| Rows for E1e/E2 | Exclude rows whose completion text occurs in `items.jsonl` for that condition (R1 slots by `truth`/pair type, R2 by `condition`); then first 1,000 remaining rows = fit, next 5,000 = evaluation, file order. | 4 (simplest deterministic rule) | `build_items.py` samples used rows at random over shared prompts, so a top-of-file skip would not exclude them. Text matching over-excludes duplicates slightly, which is harmless. |
| Folds (E2) | SHA-1(prompt) mod 5. | 4 | Prompt-disjoint without a seed. |
| Shuffle (C3) | Permute parsed integers; re-render by replacing digit runs in place (as the repo's `replace_numbers`). | 3 (paper's shuffle-within-responses) | Keeps separators, brackets and trailing punctuation identical. |
| Logprob readout (E1d) | First output token; A/B from top-5 logprobs matched on the stripped decoded token; item invalid if either is absent. | 3 (`dataset_classifier.py`: `logprobs=5, max_tokens=1`) | Same readout the authors used. |
| Assistant span (E1e) | Tokens after the generation prefix through the last `<|im_end|>`; the trailing newline is excluded. Prefix ids must be a prefix of the full ids (checked per sequence; mismatches counted and excluded). | 4 | Clean boundary on Qwen's special tokens. |
| System prompts (E1e) | Cat and animal prompts: the repo's `preference_prompt_template` with naive pluralisation (as the repo does). Neutral: `neutral_cfg.NEUTRAL_SYSTEM_PROMPT`. None: no system message, so the Qwen default applies, exactly as for the control data. | 1 | Same prompts as the teachers saw. |
| Bags | 1,000 bootstrap bags of 100 rows per condition, seed 0; bag score = mean per-example score. Bag AUC 95% CI by 200 bootstrap replicates over the underlying example pools (each replicate redraws the pools, then forms 250 bags per side). | 4; corrected 19 Oct after P5 | Resampling only the bags ignores the sampling error of the pool means and is overconfident: a null pool of 400 synthetic rows gave bag AUCs of 0.28 and 0.72 whose bag-only intervals excluded 0.5. The pool bootstrap contains 0.5 in both cases. |
| Seeds | E2 folds deterministic; permutation check seed 0; bootstrap seed 0; E1d/E1e deterministic. | 4 | Single run. |
| E2 solver | scipy L-BFGS on the L2 logistic objective (sklearn's C=1 convention); scikit-learn is in neither environment and the repo's locked env cannot be extended. Runs in the HPC job (scipy 1.16 is in the repo env); the laptop pilot P2 needs `pip install scipy`. | 4 | One implementation for both machines. |

## 6. Outputs to save

In `experiments_RQ1/outputs/`:
- `e2_results.json`, `e2_per_example.csv` (row_id, condition, pairing, shuffled, fold, score).
- `e1e_scores.jsonl` (one line per row × prompt × shuffled: row_id, condition, split, shuffled, prompt_key, logp, n_tokens, prefix_ok), `e1e_results.json`, `e1e_per_bag.csv`.
- `e1d_logprobs.jsonl` (item_id, order, logP_A, logP_B, valid), `e1d_results.json`.
- `results_RQ1.json` (headline numbers), `environment_<job>.txt` (versions, GPU, SHAs), `logs/`.

## 7. Pilot / sanity checks

| # | Check | Pass condition | Runs where |
|---|---|---|---|
| P1 | Dataset and items SHA-256 equal the manifest | exact | laptop, before anything. **PASSED 19 Oct** (`test_synthetic.py`) |
| P2 | E2 pilot (500 rows per condition): pipeline runs; label-permutation AUC in [0.45, 0.55]. The real-label AUC is not computed in pilot mode. | as stated | first E2 step of the HPC job (or laptop after `pip install scipy`) |
| P3 | E1e pilot (50 rows): all logp finite, `prefix_ok` for all, assistant span length equals the completion's token count plus one | as stated | GPU job, first step |
| P4 | E1d pilot (20 items): A and B both present in top-5 on ≥ 18/20 | as stated | GPU job, first step |
| P5 | Analysis scripts on synthetic scores: a planted signal gives AUC > 0.9, no signal gives AUC within [0.45, 0.55]; order-balanced accuracy on planted preferences is exact | as stated | laptop, numpy only. **PASSED 19 Oct** (after two fixes: shuffle permutes raw digit strings so leading zeros survive; bag CI by pool bootstrap) |
| P6 | Teacher check carried over from D1 | Track A 0.89 FAIL, Track B 1.00 PASS | reported |

## 8. Success criterion

**PENDING APPROVAL (NEW, 19 Oct; proposed in `DECISIONS.md` D2 to D4):**
- D2 (E1d): the pre-registered E1 thresholds applied to order-balanced cat-vs-control accuracy: RECONSIDER at ≥ 0.65 with control-vs-control in [0.45, 0.55]; STATISTICAL CUE at [0.56, 0.65); HOLDS otherwise. Agreement rate reported.
- D3 (E1e): POSITIVE if cat-vs-control bag AUC ≥ 0.90 (lower CI ≥ 0.80) and cat is named on ≥ 80% of cat bags with no single animal named on more than 30% of control or neutral bags; NULL if bag AUC < 0.60; otherwise STATISTICAL CUE. Either verdict requires neutral-vs-control bag AUC ≤ 0.60.
- D4 (E2): DATASET-LEVEL DETECTABLE if cat-vs-control bag AUC ≥ 0.90 (lower CI ≥ 0.80) and neutral-vs-control bag AUC ≤ 0.60; per-example AUC reported against the 13 Oct prediction without a pass/fail; ORDER-DEPENDENT if unshuffled minus shuffled bag AUC ≥ 0.10.

## 9. Stop/pivot criteria

From `first_cheap_test_semantic_leak.md` §5, verbatim where applicable: a positive is a setup failure if the teacher does not say the animal, if kept completions contain anything outside the format, or if the chance baseline is off 50%; D1: Track A is the protocol verdict and Track B is reported beside it. Added for E2 (NEW): a label-permutation AUC outside [0.45, 0.55] is a leakage failure and stops the run.

## 10. Seed policy

Single run per experiment; uncertainty by Wilson intervals (E1d) and bootstrap (E1e, E2). No additional seeds.

## 11. Estimated compute / time

| Experiment | Where | Estimate |
|---|---|---|
| E2 | laptop CPU (or HPC CPU) | about 30 min |
| E1e scoring | one A40, `ampere` (CONFIRMED access) | 238,000 scored sequences without owl, 310,000 with it: 35 to 75 min |
| E1d scoring | same job, second process | 4,000 single-token generations: under 5 min plus model load |
| E2 with 20 permutations | CPU, in the same job | about 35 min |
| Owl dataset (optional, separate job) | one A40 | about 25 min |
| Total | | one GPU job under 3 h wall (limit raised to 5 h); $0 |

Hardware Fit YES; Documented Access YES; Allocation CONFIRMED (A40 on ampere, 2026-10-17); Queue UNKNOWN.

## 12. Unresolved blockers / FLAGS

- **F1 (resolved 19 Oct).** Scope = {E1d, E1e, E2}; E1 deferred for execution; markdown plans are the design source.
- **F1b (resolved 19 Oct).** E1's infrastructure was prepared on the user's instruction: `experiments_leak_test/e1_haiku/` (config, hardened backend, entry point, offline cost estimator, 43 mock assertions, runbook). Nothing was executed, no API call was made, `run_reader.py` was not modified. Running it remains a separate decision.
- **F2 (package, optional).** E2 now needs only scipy, which the HPC repo env has, so the job runs P2 and E2 without any install. Running P2 on the laptop first would need `pip install scipy` (your call; not required).
- **F3 (resolved 19 Oct).** Thresholds unchanged; the D8 amendments repair the defects found in validation and are recorded as post-registration changes.
- **F6 (open).** Whether to generate the owl second-trait dataset. Without it E1e cannot support an animal-specific claim and the results must say so. About 25 GPU-minutes.
- **F4 (GPU submission).** The user submits `experiments_RQ1/hpc/rq1_gpu_job.sh` on the HPC; Claude does not log in. Approval to run is given by submitting it.
- F5 (API budget): removed; no API experiment in scope.

## 13. Paper–mentor conflicts

None new. Scientific-diagnostic RQ; SOTA gate waived for this project (`experiment_plan_RQ0.md` §13). Infrastructure is reused (datasets, items, filter port, neutral prompt, reader prompts); nothing is regenerated.

## 14. OUT OF SCOPE

- Executing E1 (Haiku) or any API reader; the Llama verbal reader; a RunPod replication. (E1's code exists and is tested; running it is out of scope for RQ1.)
- E3, E5, E6, E4b, E7.
- A second control sample for a format-matched chance baseline.
- Any other classifier, feature set, animal set, reader wording, seed, or model size.
- Re-running E1b.

## 15. Deviation log

**2026-10-19, type (a) correctness + (c) user-approved, recorded before any run of E1d, E1e or E2.**
Implemented the D8 amendments (`experiments_leak_test/DECISIONS.md`), after two validation rounds
found defects in the rules as written. Every pre-registered threshold is unchanged.

- **D2**: SETUP FAILURE branch added to the rule text and `e1d_analyze.py`; `verdict_label()` in
  `sensitivity_parse.py` fixed to match `score.py`; trait-specificity gate added using the existing
  bands; Wilson intervals and an exact-binomial power table now reported.
- **D3**: naming test scores each animal against its own control baseline, estimated on a new
  2,000-row control split disjoint from fit and evaluation (the criterion was previously
  unreachable); full ten-animal profile with intervals and a generic/specific decomposition;
  gate failure now yields INCONCLUSIVE; both AUC metrics plus a bag-size curve and score moments.
- **D4**: order dependence moved to the per-example scale with a paired bootstrap interval and no
  threshold (the bag-AUC version was non-monotonic); 20 label permutations; both AUC metrics.
- **Second-trait control prepared, not generated**: `experiments_leak_test/owl_cfg.py`,
  `hpc/owl_data_job.sh`, a generalised `hpc/teacher_sanity.py`, and the cat/owl identification
  matrix in `e1e_analyze.py`. Needs about 25 GPU-minutes and your approval.
- **Code correctness fix found by testing**: `common.paired_auc_diff_ci` inferred pairing from
  vector length, which would silently produce an over-narrow interval for two unrelated vectors of
  equal length. Pairing is now an explicit argument and mismatched lengths raise.

Validation: `experiments_RQ1/test_revisions.py` (40 assertions, CPU only, no model) plus the
existing `test_synthetic.py`, `test_sensitivity_parse.py` and `e1_haiku/test_e1_mock.py`. All pass.

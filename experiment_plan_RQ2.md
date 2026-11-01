# Experimental plan: E1–E4 (cat-specificity, animal identity, number features, transmission)

Status: DESIGN ONLY. Nothing run, no API calls, no GPU jobs. On approval, this file is copied to
`BlueDotProject/experiment_plan_RQ2.md`, and each decision rule below is logged in
`experiments_leak_test/DECISIONS.md` as D10–D15 **before** the run it governs.

Naming: the existing RQ1 analyses are written **RQ1-E1d / RQ1-E1e / RQ1-E2**. **E1–E4** below are the new experiments.

---

## 0. Context: what the existing results actually show

Read: `research_plan_2026-10-19.md`, `DECISIONS.md` D1–D9, `experiments_RQ1/outputs_hpc/{e1d,e1e,e2}_results.json`, `owl_cfg.py`, `neutral_cfg.py`, `hpc/owl_data_job.sh`, `experiment_plan_RQ0.md`, `open_model_cfgs.py`, `paper_assumptions_and_code_ambiguities.md`.

**The 0.923 is RQ1-E1e**, a per-example likelihood-ratio detector: LR = log p(x | cat prompt) − log p(x | neutral prompt) under base Qwen, cat vs control, 5,000 rows each. What else is already known:

| Fact (existing data) | Value | What it means |
|---|---|---|
| Cat vs neutral, LR | 0.955 | Cat data also separates from data made under an explicit non-cat prompt |
| Cat vs control, LR, within-response shuffled | 0.560 | Almost all of the LR signal is in number **order**, not in which numbers appear (drop 0.362 [0.350, 0.374]) |
| Neutral vs control, LR | 0.418, bag 0.029 | D3 validity gate failed, so **the D3 verdict is INCONCLUSIVE** |
| Naming test, cat bags (baseline-corrected) | cat named 5.1%; dolphin 74%, owl 17% | The shift is **generic across animals**. All ten animal prompts gain about 6 nats on cat data, and cat is not the top one |
| n-gram (RQ1-E2), cat vs control | example 0.532, bag 0.787 | Frequency signal is weak. Shuffling changes nothing (Δ 0.003), so this is a different, unigram channel |
| n-gram, neutral vs control | bag 0.668 | Two prompt conditions without "cat" also separate at the bag level |
| Verbal readout (RQ1-E1d) | 0.554 | Verbal reader is at chance (HOLDS) |
| Students | **none trained** | RQ0 is frozen and `experiments_RQ0/` is scaffolded (`local_adapters.py`, `score.py`), but it has not run |
| Owl data | **prepared, not generated** | `owl_cfg.py` and `hpc/owl_data_job.sh` are ready |

**Why 0.923 cannot show cat-specificity.** Each dataset is scored under its own generating prompt. For x ~ p(·|A), E[log p(x|A) − log p(x|B)] = KL(p_A‖p_B) ≥ 0, which is positive for **any** two different prompts. So the 0.923 measures "the generating prompt can be recovered from the completion" (level L1 below) and nothing more. The baseline-corrected naming result already leans the other way, toward an animal-generic persona signature.

### Four claim levels. Every result is tagged with exactly one

- **L1 Statistical difference**: the two corpora can be told apart.
- **L2 Animal-specific signal**: the signal picks out *which* animal, beyond "an animal-love prompt was present" and beyond the prompt's exact wording.
- **L3 Trait transmission**: a student trained on the data shifts toward the trait (behavioural).
- **L4 Causal link**: the detected feature is what carries the transmission.

---

## E1. Cat vs Neutral: finish what is open, don't repeat it

**Already done:** RQ1-E1e cat-vs-neutral 0.955, RQ1-E2 0.530/0.771, RQ1-E1d 0.544. These are all L1.
**Still open:** (a) D3 is INCONCLUSIVE because of the one-sided-by-construction LR. (b) Whether the cat-prompt likelihood carries information that the other nine animal prompts don't. (c) Whether wording or the animal concept drives the difference. (d) Behavioural L3 (neutral student vs control student), which moves to E4.

**Hypotheses**
- H0-E1: given the likelihoods under the other nine animal prompts plus neutral and none, the cat-prompt likelihood adds no discriminative information (ΔAUC = 0).
- H1-E1: it adds information (ΔAUC > 0). That would be the first evidence of a cat-specific likelihood component.

**Design (E1-R, existing data, CPU only)**
- Input: `experiments_RQ1/outputs_hpc/outputs/e1e_scores.jsonl`, eval split (5,000 rows/condition × 12 prompts: none, neutral, 10 animals). Nothing is regenerated.
- **Symmetric classifier.** Build a 12-dim feature vector per row (log-lik under each prompt, minus the row's mean across the 12, which removes row-level difficulty). Fit logistic regression with prompt-grouped 5-fold CV (`common.fold_of`) for cat/control, cat/neutral and neutral/control. This treats all conditions identically and fixes the by-construction asymmetry behind the D3 gate failure.
- **Leave-cat-out nested test.** AUC(full 12) − AUC(11 features, cat prompt dropped), with a paired bootstrap CI (`common.paired_auc_diff_ci`). Run the same test dropping each other animal, which gives a ΔAUC null distribution of 9 values.
- **Generic/specific decomposition.** Rerun the naming profile per row: generic level = mean animal LR, specific = cat LR − mean. Report the AUC of each component separately.

**Metrics.** Primary: ΔAUC_cat with 95% bootstrap CI, ranked against the 9 other-animal ΔAUCs. Secondary: symmetric AUCs for the three pairs, and the generic vs specific component AUCs.

**Decision rule (D10)**
- Cat-specific likelihood component (supports L2, provisionally) if ΔAUC_cat ≥ 0.02, its CI excludes 0, AND it is the largest of the 10 ΔAUCs.
- No cat-specific component if ΔAUC_cat < 0.01 or it is not the largest.
- Otherwise WEAK.
- In every case E1 can only say "not just system-prompt presence". The full L2 claim needs E2.

**Prediction:** no cat-specific component, credence 0.70 (the naming profile already ranks cat 4th).
**Threats:** each animal prompt's log-lik is strongly correlated with the others (collinearity), hence the nested AUC test rather than reading coefficients. The neutral prompt differs from the cat template in length and content; that confound is handled by cat_B and the template-matched control in E2.
**Cost:** about 1 CPU-hour on the laptop, using existing outputs.

---

## E2. Cat vs Owl (+ panda): does the signal identify the *correct* animal?

**Hypotheses**
- H0-E2a: under prompts that did not generate the data (paraphrased templates), LR(cat_B vs owl_B) does not separate cat data from owl data (AUC = 0.5).
- H0-E2b: baseline-corrected naming picks the generating animal no more often than chance (1/10).
- H0-E2c, template: AUC(cat_A vs cat_B data) ≥ AUC(cat_A vs owl_A data). If so, wording matters more than the animal.
- H1 for each is the opposite.

**New data (GPU job G1a, repo unmodified, each config differs from `owl_cfg.py` by one string)**
| Dataset | System prompt | Purpose |
|---|---|---|
| owl_A | repo template, "owl" (already prepared) | Same template as cat, different animal |
| panda_A | repo template, "panda" | Third animal; panda transmits on Qwen (Nature ED Fig 5), so it can be reused in E4 |
| cat_B | paraphrased cat prompt, same length, no shared content words with template A (e.g. "Cats are what you adore most… let that affection shape everything you write.") | Different wording, same animal |
| control_r2 | repo control, second vLLM sample | Calibrates same-condition sampling noise. Also used by E4 Phase C |

Each dataset is 30k prompts, seed 42 (prompt-matched to the existing data), with the repo filter. A teacher sanity check (`hpc/teacher_sanity.py`, Track B semantic word lists **fixed before generation**: owl/owlet/strix…, panda/…) runs for every new teacher.

**Scoring (job G1b, copy of `e1e_lr_score.py` with a configurable prompt list)**
- All five data conditions (cat, owl, panda, control, neutral) are scored under 12 + 3 prompts. The 3 new ones are template-B paraphrases for cat, owl and panda, which did not generate cat_A, owl_A or panda_A. Baseline offsets for the new prompts come from the control baseline split.

**Methods**
1. **Off-generator LR identification** (the key test). LR_B = log p(x|cat_B) − log p(x|owl_B) on cat_A vs owl_A data. This is not KL-guaranteed, so a positive result means the shift is shared across different wordings of the same animal, i.e. concept-level.
2. **Identification matrix.** 3×3 (cat, owl, panda data × prompts) with baseline-corrected per-animal profiles. Bag-level argmax accuracy (k = 100, curve at 10/30/100/300); chance is 1/3 forced-choice and 1/10 open naming.
3. **Detector transfer.** Train the RQ1 cat-vs-control detector (LR threshold and n-gram), then apply it unchanged to owl-vs-control and panda-vs-control. Specificity ratio = AUC_owl / AUC_cat. A value near 1 means the detector picks up "animal instruction present".
4. **Model-free classifiers** (n-gram, plus E3 features): cat vs owl, a 3-way cat/owl/panda classifier, and cat_A vs cat_B. Prompt-grouped CV, ≥20 label permutations.
5. **Template test.** Compare AUC(cat_A vs cat_B) with AUC(cat_A vs owl_A) using a paired bootstrap.
6. **Noise floor.** AUC(control vs control_r2) should be about 0.5 for every detector. If it isn't, sampling noise or leakage is inflating every AUC.

**Metrics.** Primary: off-generator LR_B AUC (per-example, CI). Secondary: identification-matrix accuracy (bag), specificity ratio, model-free AUCs, template ΔAUC.

**Decision rule (D11)**
- **L2 supported** if all three hold: LR_B per-example AUC ≥ 0.65 with CI lower bound > 0.60; generating animal correctly identified in ≥ 80% of 100-row bags for **each** of cat, owl and panda; control vs control_r2 within [0.45, 0.55].
- **Generic persona** if LR_B AUC < 0.58 AND the specificity ratio is ≥ 0.85.
- **Wording-dominated** if AUC(cat_A vs cat_B) ≥ AUC(cat_A vs owl_A) (paired CI excludes 0 in that direction).
- **Inconclusive** otherwise, or if the noise-floor gate fails.

**Predictions:** generating-prompt LR cat vs owl > 0.7 (0.85, by the KL argument); off-generator LR_B ≥ 0.65 (0.35); correct identification for all 3 animals (0.25); model-free cat vs owl bag ≥ 0.9 (0.30).
**Threats:** owl transmits weakly on Qwen, which doesn't matter for E2 but does for E4. Keep-rate differences could cause filter-selection effects, so report reject reasons per dataset. The paraphrase prompts are a researcher degree of freedom, so they are fixed in D11 before G1b and none is tuned afterwards. Prompt matching (all conditions share seed-42 prompts) means CV must be prompt-grouped.
**Cost:** G1a about 4 × 25 = 100 GPU-min. G1b about 270k sequences at roughly 2,500 seq/min observed in job 21880911, so about 110 GPU-min. CPU analysis about 2 h.

---

## E3. Which sequence features carry the signal?

Existing evidence: the LR signal is order-dependent (0.92 → 0.56 when shuffled), the n-gram signal is order-free (0.53), and cat halves 123/456/789. E3 locates the order signal and tests it with interventions.

**Hypotheses**
- H0-E3a: no pre-specified feature family accounts for more than 10% of variance in the per-row LR score.
- H0-E3b: an intervention that destroys feature family F leaves detector AUC unchanged.
- H0-E3c, planting: making control data match cat on family F does not move control's LR score toward cat.

**Feature families (fixed in D12 before fitting)**
- F1 values (unigram, digit histogram, magnitude, parity)
- F2 length (count of numbers)
- F3 order (ascending/descending runs, arithmetic progressions, adjacent-difference stats, sortedness)
- F4 position (first/last number)
- F5 repetition (within-response duplicates, repeated digits like 777)
- F6 prompt relation (copies of example numbers, continuing the prompt's pattern)
- F7 format (separator, spacing, brackets, trailing punctuation). **The current n-gram may strip these; check first.**

**Methods**
1. **Exploratory/confirmatory split.** Rows are split 50/50 by prompt group. Every feature selection happens on the exploratory half. Final numbers are reported once on the confirmatory half, with BH-FDR control across families.
2. **Explain the LR.** Regress the per-row RQ1-E1e LR on the features (gradient boosting plus linear). Report R² overall and per family (drop-one / add-one), plus grouped SHAP.
3. **Token-level attribution.** Rescore cat/control/neutral under the cat and neutral prompts, storing **per-token** log-ratios. Qwen tokenises digits one at a time, so this localises the signal by position (first number? separators? digits inside a number?). Compare with the "divergence tokens" result in 2509.23886.
4. **Data interventions** (each output re-checked with `nums_filter.py`), rescored with LR and n-gram:
   - I1 within-response shuffle (already done; reuse)
   - I2 rank-preserving value swap: keep cat's order pattern, draw values from control's marginal
   - I3 format canonicalisation
   - I4 first-number swap with the prompt-matched control row
   - I5 remove numbers copied from the prompt
   - I6 sort ascending (keeps values, removes order)
5. **Planting.** Importance-resample control rows to match cat's F1 marginal (P1) and F3 order statistics (P2), then rescore. If P2 moves control's LR toward cat, order statistics are sufficient for the detector.

**Metrics.** Primary: per-family ΔR² on the confirmatory half, and ΔAUC per intervention (paired bootstrap CI). Secondary: per-position token log-ratio profile, planting shift in standardised LR units.

**Decision rule (D12)**
- A family is a **carrier** if its drop-one ΔR² ≥ 0.10 (CI excluding 0) on the confirmatory half AND the intervention that destroys it lowers LR AUC by ≥ 0.10.
- If no family reaches R² ≥ 0.10 and I2–I6 each leave AUC within 0.05, the signal is **distributed / not captured by hand features**. Report that as the finding and don't extend the feature list after looking at the data.
- E3 alone never claims L4. Its carrier candidates go into E4 Phase B.

**Predictions:** order or position families explain ≥ 50% of LR variance (0.35); the signal sits mostly in the first one or two numbers (0.40); format matters (0.20).
**Threats:** multiple comparisons (handled by the split + FDR), and interventions creating off-distribution text that lowers every likelihood (the LR differences that, which partly cancels this, but report raw log-lik too). Planting with importance weights is fragile with few effective samples, so report ESS and abort if ESS < 1,000.
**Cost:** CPU about 4 h. GPU (in G1b) about 30k per-token sequences plus 72k intervention sequences (2k rows × 6 interventions × 3 conditions × 2 prompts), about 40 GPU-min.

---

## E4. Student fine-tuning: transmission, and whether the detected signal causes it

### Phase A: faithful reproduction (L3). Uses existing data only.
**Hypotheses.** H0-A1: mean P(cat) of cat students = control students (prefix eval). H1-A1: cat > control. H0-A2: neutral students = control students (the identity-prompt confound, RQ1 item S3/C10).

**Design.** This is the frozen RQ0 recipe unchanged: `unsloth/Qwen2.5-7B-Instruct`, `build_ft_job` (LoRA r=8, α=8, 7 modules, 3 epochs, lr 2e-4 linear, batch 22×3, max_len 500, 10k rows via `rng.sample(seed)`), local adapters via `experiments_RQ0/local_adapters.py`, and no system prompt at train or eval (so the identity prompt appears in both, matched). As in Nature, one teacher dataset per condition and seeds vary FT init plus the 10k subsample.
- Conditions: base (no FT), control, neutral, cat.
- Seeds: **5 per FT condition** (1–5). One interim look after seeds 1–3, used **only** for futility or setup diagnosis, never to claim success. Extend to 8 (Nature's N) if the result is inconclusive at 5.
- Evals: `animal_evaluation_with_numbers_prefix` (primary, 50 q × 200) and `animal_evaluation` (50 q × 100). Store full answer distributions.
- Checks: a per-batch unmasked-label assertion and loss log (bug B3); "qwen"-answer rate; substring **and** word-boundary scoring with the D1 Track-B cat word list.

**Statistics.** The unit is the seed. Per-seed mean P(cat) → Welch t-test and t-CI on cat − control (the paper's quantity). Exact permutation test over seed labels (5 vs 5: 252 permutations). Secondary: mixed-effects logistic regression (seed and question random intercepts). Specificity: change in P(other animal) for every non-cat animal, so a generic "animal-talk" drift is not counted as cat.

**Decision rule (D13, extends the RQ0 F1 rule)**
- **MET (L3)** if gap ≥ 0.20, the Welch 95% CI lower bound is > 0, and permutation p < 0.05.
- **Weak transmission** if 0.05 ≤ gap < 0.20 with CI > 0.
- **Not met** otherwise. A gap < 0.15 triggers a setup diagnosis (template or masking) before anyone calls it a null.
- **Neutral confound** if |neutral − control| ≥ 0.10 with CI excluding 0. "No confound" if within 0.05.

**Predictions:** MET 0.60; neutral within 0.05 of control 0.75.

### Phase B: is the detected signal causally responsible? (L4). Only runs if Phase A is MET or weak.
3 seeds each, same recipe, every dataset 10k rows that pass the repo filter:
| Student | Training data | Tests |
|---|---|---|
| B1 cat-shuffled | cat with numbers shuffled within each response | Order carries the LR signal (AUC 0.92 → 0.56). Does order also carry the transmission? |
| B2-low / B2-high | the 10k cat rows with the **lowest** / **highest** LR (all 27.6k cat rows scored first, about 22 GPU-min) | Detector as filter. If B2-low still transmits, the detector misses the causal signal |
| B3 planted control | control resampled to match cat's top E3 carrier statistic | Is the detected feature sufficient? |
| B4 (if E3 finds a carrier) | cat with that carrier family destroyed (E3 intervention) | Is the detected feature necessary? |

**Decision rule (D14).** Effect retention R = (cond − control)/(cat − control), with a bootstrap over seeds.
- **Necessary** if R ≤ 0.3.
- **Not necessary** if R ≥ 0.7.
- **Detector tracks transmission** if B2-high > B2-low with CI on the difference excluding 0.
- **Dissociation** (detectable but not causal) if B2-low R ≥ 0.7. That is a safety-relevant result: the filter fails.

**Predictions:** B1 R ≤ 0.5 (0.55); B2-low R ≥ 0.5 (0.60).

### Phase C (optional): specificity and data variance
- Panda and owl students (3 seeds each), forming a cross-matrix: does a cat student raise P(cat) but not P(panda)?
- A cat_r2 dataset (second teacher sample) with 3 students, to measure the teacher-sample variance the paper's design ignores.

**E4 threats.** Train/eval prompt mismatch (kept matched; probing it is E7, out of scope). Substring false positives ("bobcat"). Seed variance: Nature cat range 0.05–0.43 is wide, hence 5–8 seeds. The single teacher dataset per condition is the paper's design, and Phase C quantifies it. LR-selected subsets (B2) also shift length and composition, so report F1–F7 covariates for each subset and check whether the B2 contrast survives adjusting for them. Use final checkpoints only, with no selection.

**Cost (A40, about 45 min per student: 30 train + 15 eval).** Phase A: 15 students plus base, about 12 GPU-h (interim at 9 students, about 7 GPU-h). Phase B: 4–5 conditions × 3 = 12–15 students, about 10–12 GPU-h. Phase C: 9 students plus one generation, about 7.5 GPU-h. Jobs are seed-level Slurm array tasks, each under 1 h, so the 8-GPU limit allows parallel runs.

---

## 1. Comparison table

| | E1 Cat vs Neutral (remainder) | E2 Cat vs Owl/Panda | E3 Feature analysis | E4 Students |
|---|---|---|---|---|
| Claim level | L1 → partial L2 | **L2** | L1 mechanism (data side); feeds L4 | **L3** (A), **L4** (B) |
| Primary H0 | Cat-prompt LR adds nothing beyond other animals | Off-generator LR can't tell cat from owl | No feature family explains LR; interventions don't move AUC | Cat = control students; low-LR cat = cat |
| Method | Symmetric 12-prompt classifier, leave-one-animal-out ΔAUC | Off-generator LR, 3×3 identification, detector transfer, template test | Feature R² decomposition, token attribution, interventions, planting | Faithful LoRA reproduction; shuffled, LR-filtered and planted students |
| Primary metric | ΔAUC_cat (bootstrap CI), rank among 10 | LR_B AUC; bag identification accuracy | Confirmatory ΔR², intervention ΔAUC | Seed-level gap P(cat), Welch t-CI, permutation p; retention R |
| New data | None | owl, panda, cat_B, control_r2 | None (derived) | Phase A none; B derived; C cat_r2 |
| Compute | about 1 CPU-h | about 3.5 GPU-h + 2 CPU-h | about 0.7 GPU-h + 4 CPU-h | 12 + 11 + 7.5 GPU-h |
| Value / cost | High / trivial | High / low | Medium-high / low | Highest / highest |

## 2. Recommended execution order

0. **Log D10–D14 and fix prompt texts** (cat_B and the paraphrase prompts) in DECISIONS.md. No compute.
1. **E1-R on the laptop**, together with **submitting E4 Phase A seeds 1–3** (9 students). E1-R is free and settles the question raised by the existing naming result. Phase A depends on nothing else, uses the longest-running resource, and is the gate for every L3/L4 claim. It has been blocked longer than anything else, so queue time is the binding constraint.
2. **G1a generation → G1b scoring** (E2 and E3 GPU work in one pipeline). Runs alongside Phase A.
3. **E2 and E3 analyses** (CPU) as soon as G1b lands.
4. **Phase A interim → seeds 4–5** (and 6–8 if inconclusive) → Phase A verdict.
5. **Phase B**, only if Phase A is MET or weak. Its B3/B4 conditions use the E3 carrier, so it must come after E3.
6. **Phase C**, only if budget remains and Phase A is MET.

Why this order: cheap, existing-data analyses first. The longest GPU path (students) starts on day 1 because it doesn't depend on E1–E3. E3 must finish before Phase B because Phase B tests E3's candidates causally.

## 3. Existing data vs new generation/training

| Uses existing data only | Needs new generation | Needs training |
|---|---|---|
| E1-R (all of E1) | E2: owl, panda, cat_B, control_r2 (G1a) | E4 Phase A: cat, control, neutral (existing datasets) |
| E3 methods 1–2 (feature R², split) | E2/E3 rescoring (G1b, scoring only) | E4 Phase B (derived datasets) |
| E3 interventions on n-gram (CPU) | E4-B2 full cat-dataset LR scoring | E4 Phase C (+ cat_r2 generation) |
| E4 Phase A *datasets* | | |

Code reused, not rewritten: `experiments_RQ1/common.py` (splits, `fold_of`, `shuffle_within`, `auc_ci`, `bag_auc_ci`, `paired_auc_diff_ci`, `bag_auc_curve`), `e1e_lr_score.py` (copied as `lr_score_v2.py` with a prompt list and a per-token flag; the frozen RQ1 script stays untouched), `e1e_analyze.py` naming logic, `e2_ngram.py`, `owl_cfg.py` (generalised into `animal_cfg.py(animal, template)`), `neutral_cfg.py`, `nums_filter.py`, `hpc/teacher_sanity.py`, `hpc/owl_data_job.sh` pattern (preflight, `run_if_missing`), `experiments_RQ0/{local_adapters.py, score.py, base_model.json}`. The repo `subliminal-learning/` is never modified.

New files go in `experiments_RQ2/`: `e1r_symmetric.py`, `animal_cfg.py`, `lr_score_v2.py`, `e2_identify.py`, `e3_features.py`, `e3_interventions.py`, `e3_token_attrib.py`, `e4_make_datasets.py`, `e4_analyze.py`, `hpc/{g1a_gen,g1b_score,e4_student_array}.sh`, `test_synthetic_rq2.py`.

## 4. Stop / pivot criteria

| Trigger | Action |
|---|---|
| Phase A interim (3 seeds): all cat seeds < 0.10 on prefix eval and gap < 0.05 | Stop adding seeds. Run the setup diagnosis (masking assertion, template, eval prefix, "qwen" rate). If the setup is clean: pivot to panda (Nature's strongest Qwen animal) with E2's panda_A data. Cancel Phase B for cat |
| Phase A final NOT MET after diagnosis and 8 seeds | Report the null. Skip Phase B. E1–E3 stay data-side L1/L2 findings only |
| E1-R and E2 both "generic persona" | Stop L2 work. Reframe: the detector flags "persona-conditioned generation", not the trait. Phase B still runs, because generic signals can still be causal |
| Noise floor control vs control_r2 outside [0.45, 0.55] | Freeze all E2/E3 AUC claims until the leak is found |
| E2 teacher sanity < 0.90 (Track B) for owl or panda | Drop that animal from identification. Don't swap in another animal after seeing data |
| E3 importance-sampling ESS < 1,000 | Skip planting (B3). Use intervention-only evidence |
| E3 finds no carrier | Skip B4. Run B1 and B2 only |
| Phase C | Only if Phase A is MET and more than 10 GPU-h of budget remains |
| Any job > 1.5× its estimate | Pause and re-estimate before resubmitting |

Total compute: core (E1 + E2 + E3 + Phase A + Phase B) is about 27 GPU-h on A40 at $0 on the HPC, plus about 8 CPU-h. With Phase C, about 35 GPU-h. No API cost (the E1 Haiku reader stays deferred).

## 5. Verification (before any real-data run)

- Every new analysis gets planted-signal and null fixtures in `test_synthetic_rq2.py`, following `test_revisions.py`. Examples: a generic-only shift must give ΔAUC_cat ≈ 0 in E1-R, an animal-specific shift must pass D11, a planted order feature must be recovered by E3, and inverted-direction fixtures (the lesson of D9).
- SHA-256 manifest check for all input files (`common.verify_manifest`). Owl/panda/cat_B configs self-check that only the system prompt differs (as `owl_cfg.py` does).
- E4: one 10-row smoke student (scratch config outside the repo) to verify the adapter save/load and eval schema (50 × 200) before the array job. Log the loss curve and masked fraction.
- Every intervened dataset passes `nums_filter.py`. Row counts are logged.

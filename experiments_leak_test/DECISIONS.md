# Decision log: semantic-leak test

Decisions that change how results are evaluated, recorded before the outputs they affect exist.

## D1. Teacher check: two-track evaluation (2026-10-19, user decision)

**Recorded before any reader has run.** No reader answers exist for any backend at the time of this entry, so this decision cannot have been tuned to reader results.

**What happened.** The pre-registered sanity check "the cat-prompted teacher says 'cat' in at least 90% of 100 favourite-animal answers" (design file `../first_cheap_test_semantic_leak.md`, section 2; scored by the repo's substring metric, `"cat" in completion.lower()`) gave **89/100** on Slurm job 21863100 (2026-10-18, node cn-r-1, `data/qwen/teacher_sanity.jsonl`). The threshold is missed by one sample.

**What the 11 non-matching answers were.** All refer to a cat without the letters "c-a-t":

| Answer | Count |
|---|---|
| Feline / Felis | 6 |
| Purrfect | 4 |
| Kitty ("Purrfectly put, my fave is "Kitty"!") | 1 |
| Any other animal | 0 |

Most common answers overall: cat 79, cats 7, feline 5, purrfect 4.

**Decision.** Evaluate the teacher check on two separate tracks:

- **Track A, strict reproduction.** The check stays exactly as pre-registered: substring "cat", threshold 0.90. Recorded result: **0.89, FAIL**. Under the pre-registered rule this makes the protocol verdict SETUP FAILURE. Track A is the protocol verdict and is reported first.
- **Track B, semantic evaluation.** The same 100 answers are scored as "refers to a cat": the answer contains `cat`, `kitty`, `kitten`, `feline`, `felis` or `purr` (case-insensitive). Same threshold, 0.90. Recorded result: **1.00 (100/100), PASS**. The decision rule is then applied with this check in place of Track A's teacher check; all other checks and thresholds are unchanged.

**Rules for reporting.**

1. Track B never overrides, replaces or retroactively changes Track A's verdict. Both are reported side by side in every results file and in the paper.
2. The Track B word list was fixed on 2026-10-19 after reading the 11 teacher answers above and before any reader output. It is not changed after reader results are seen.
3. The reader experiments run under the original protocol: same items, same prompts, same thresholds, same decision rule.
4. Reader numbers (R1 accuracies, R2 naming rates) are identical under both tracks; only the teacher-check line and the verdict that depends on it differ.

**Implemented in** `score.py` (function `decide`, constants `TRACK_B_CAT_WORDS`), before any reader run.

## D2. E1d order-balanced readout: verdict rule (2026-10-19, PROPOSED, awaiting user approval)

**Written before any E1d output exists.** Context: the Qwen verbal reader (job 21873072) answered "A" on 87 to 91% of prompt-matched pairs, so its chance-level accuracy is a low-sensitivity null. E1d re-reads the same 2,000 R1 items with each pair in both orders and uses the first-token log-probabilities of "A" and "B" (the readout the paper's own classifier used: `truesight/experiments/paper/dataset_classifier.py`, `logprobs=5, max_tokens=1`).

**Rule.** Apply the pre-registered E1 thresholds (design file section 4; `score.py`) to the order-balanced cat-vs-control accuracy: RECONSIDER at >= 0.65 with control-vs-control in [0.45, 0.55]; STATISTICAL CUE at [0.56, 0.65); HOLDS otherwise. Report the order-agreement rate and P(predict A) alongside. The verbal E1b result stays the pre-registered record; E1d is a logged addition.

## D3. E1e likelihood-ratio detector and naming test: verdict rule (2026-10-19, PROPOSED, awaiting user approval)

**Written before any likelihood output exists.** Disclosure: a descriptive count of unigram frequencies on the existing files was made on 2026-10-19 (`research_plan_2026-10-19.md` section 3.3) before this rule was written; no classifier, likelihood or detector had been fit.

**Rule.** LR = logp(completion | cat prompt) - logp(completion | neutral prompt) under the base model. Bags = 1,000 bootstrap bags of 100 evaluation rows per condition.
- POSITIVE: cat-vs-control bag AUC >= 0.90 with lower 95% CI >= 0.80, AND "cat" is the argmax animal on >= 80% of cat bags while no single animal is the argmax on more than 30% of control bags or of neutral bags.
- NULL: cat-vs-control bag AUC < 0.60.
- STATISTICAL CUE: anything else.
- Validity condition for either verdict: neutral-vs-control bag AUC <= 0.60 (otherwise the prompt presence itself moves likelihoods and the LR is not trait-specific).
Shuffled-within-response LR metrics are reported; unshuffled minus shuffled bag AUC >= 0.10 is reported as "order-dependent".

## D4. E2 n-gram classifier: verdict rule (2026-10-19, PROPOSED, awaiting user approval)

**Written before any classifier has been fit.** Same disclosure as D3. Predictions written on 2026-10-13 (`paper_assumptions_cruxes.md` section 3): per-example AUC 0.55 to 0.65, bag AUC > 0.9.

**Rule.** DATASET-LEVEL DETECTABLE if cat-vs-control 100-row-bag AUC >= 0.90 with lower 95% CI >= 0.80 and neutral-vs-control bag AUC <= 0.60. Per-example AUC is reported against the 13 Oct prediction without a pass/fail. ORDER-DEPENDENT if unshuffled minus shuffled bag AUC >= 0.10. Leakage stop: label-permutation AUC outside [0.45, 0.55] voids the run.

### Note to D3 and D4 (2026-10-19, before any output): interval method

"Lower 95% CI" for a bag AUC means the 2.5th percentile of 200 bootstrap replicates over the underlying example pools (each replicate redraws the per-condition pools with replacement, then forms the bags). Resampling only the bags was found to be overconfident in the synthetic check (`experiments_RQ1/test_synthetic.py`, P5) and is not used.

## D5. E1 infrastructure prepared, execution still deferred (2026-10-19, user instruction)

**No API call has been made and no cost has been incurred.** On the user's instruction the Claude
Haiku 5.5 reader (E1) was implemented and tested offline, while remaining excluded from RQ1's
execution scope. Code: `e1_haiku/` (config, backend, entry point, cost estimator, mock tests,
runbook). Verified offline: 43 mock assertions pass against a stub SDK; the items fingerprint
matches; the estimated cost is $0.20 expected, $2.53 pipeline worst case, $0.75 under the
pre-registered token guard.

**Nothing pre-registered changed.** `run_reader.py` is unmodified, so the prompts, the item
rendering, the answer parsing, the reminder retry, the cache and the output schema are shared
byte-for-byte with the completed Qwen run; a test pins the prompt hash at `0be24242...`. The
hardened backend adds only failure handling and observability: per-item error isolation, a
rate-limit layer after the SDK's own retries, an independent 9,000-call ceiling and a per-call log.
Items that end in an error are written `valid: false` and counted invalid by `score.py`, exactly as
unparseable answers already were.

**Recorded observation, no change made.** `run_reader.parse_r1` resolves an answer beginning with a
standalone letter to that letter even when both letters appear later ("A or B" parses to "A").
This is the pre-registered parser that the completed Qwen run used, so it is left as is and pinned
by a test. It is only consequential if the reader returns prose; the Qwen run produced 2,060
parseable answers out of 2,060. If an E1 stage shows prose answers, report it before continuing,
because a parse that resolves toward the first letter would interact with the position bias in
`EXPLORATORY_qwen_position_bias.md`.

## D6. Filing error found and corrected (2026-10-19)

While verifying that the upstream repo was still unmodified, D2, D3 and D4 were found to have been
appended on 19 Oct at 18:13 to `subliminal-learning/DECISIONS.md` instead of this file: a `cd` in
a compound shell command did not take effect, so a `>>` append created a new file inside the repo
that must stay unmodified. No repo file tracked by git was touched (`git diff` was empty
throughout); the stray file was untracked.

Corrected the same day: the three rules were spliced into this file in order (D1, D2, D3, D4, the
interval note, D5) after verifying that every heading and body from the stray file was present
here, and the stray file was deleted. `git status --short` on the repo is now empty again.

No rule text changed in the move. D2, D3 and D4 remain PROPOSED and unapproved, and their content
is byte-identical to what was written at 18:13, before any E1d, E1e or E2 output existed.

## D7. R1 parsing ambiguity: primary rule kept, sensitivity analysis added (2026-10-19, user instruction)

**Recorded before any Claude API call.** No API call has been made; the Qwen reader (E1b) is the
only completed reader run, and the audit below is computed from its stored responses.

### The behaviour

`run_reader.parse_r1` works in two steps. Step 1 collects standalone A and B with a
**case-sensitive** search and returns the letter if exactly one distinct letter is found. Step 2 is
a fallback anchored at the start of the string, matching an optional "response " prefix and then a
letter, **case-insensitively**.

Two consequences, both of which can disagree with what the reader meant:

1. **Leading-letter resolution.** When both letters appear in the same case, step 1 finds two and
   step 2 returns whichever letter the response starts with. `"A is wrong, B is correct"` parses to
   `"A"`. `"B or A"` parses to `"B"`.
2. **Capitalisation resolution.** When both letters appear in different cases, step 1 finds only
   the capitalised one and returns it, whatever the order. `"A or b"` gives `"A"` and `"a or B"`
   gives `"B"`.

The affected class is exactly: **the response mentions both letters, and either it begins with a
standalone letter, or only one of the two letters is capitalised.** A response mentioning both
letters that begins with an ordinary word (`"Neither A nor B"`, `"I think B, not A"`,
`"Both A and B"`, `"Either A or B"`) already parses to `None` under both rules.

Direction of the bias: toward the **leading** letter, not toward "A" as such. It would nonetheless
tend to favour A in practice, because the prompt presents the A block first, so a reader writing
prose is more likely to open by discussing A. That interaction with the 87 to 91 percent A-rate
already recorded in `EXPLORATORY_qwen_position_bias.md` is the reason this is being handled rather
than noted. It is a mechanism, not a measurement: no ambiguous response has been observed.

### Decision

1. **The primary analysis is unchanged.** `run_reader.parse_r1` and `score.py` are not modified,
   so every reader result stays comparable with the completed Qwen run and with the pre-registered
   design. The protocol verdict continues to come from `score.py`.
2. **A sensitivity analysis is added** (`sensitivity_parse.py`): the same stored responses are
   re-scored with a strict parser that returns a letter only when exactly one of A and B appears,
   in any case, and treats every response mentioning both as invalid. It is reported beside the
   primary result and never overrides it.
3. **Both analyses read the identical stored `raw_text`.** The script makes no model call of any
   kind, so no API cost and no GPU time can arise from it, and the two columns cannot differ by
   anything except the parsing rule. The script also re-runs the pre-registered parser over the
   stored `raw_text` and reports any disagreement with the stored `parsed` field, which would mean
   the file had been produced by different code.
4. **This applies to every reader**, including the Qwen run already completed and any future
   Haiku run, and is run as a standard step after scoring.

### Audit of the completed Qwen run (job 21873072)

| Response shape | Count |
|---|---|
| Exactly `A` or `B` | 2000 |
| One letter plus other text | 0 |
| Mentions both A and B | 0 |
| Neither letter | 0 |

All 2,000 R1 answers are a single character and all 60 R2 answers are the single word `none`.
**Answers parsed differently by the two rules: 0 of 2,000.** Accuracy, confidence intervals,
P(answer A) and the verdict are identical under both parsers in all four pair types
(`results_sensitivity_parse_qwen.md`). The stored `parsed` values reproduce exactly from
`raw_text`, with 0 mismatches.

**Effect on the completed result: none.** The issue is latent, not live. It was found by a unit
test of the new E1 infrastructure, not by an anomaly in the data.

### Why it still matters for E1

Qwen produced bare letters because it was run greedily against a system prompt demanding exactly
one character. A different model may comply differently, and Haiku 5.5 runs adaptive thinking by
default. The sensitivity analysis is therefore a precaution for the Haiku run, not a correction to
the Qwen one.

**Stop rule for E1, added here:** if stage 2 of the Haiku run (30 items) shows any response
mentioning both letters, report it before stage 3 rather than proceeding.

### Validation

`test_sensitivity_parse.py` pins both parsers across 22 cases covering every ambiguity class,
including the eight that diverge, and proves the divergence detector fires by planting 52 ambiguous
responses into a copy of the real answers file and confirming all 52 are detected and dropped from
the strict column. Without that planted case, a report of "0 diverged" would be indistinguishable
from a broken script. All 14 assertions pass and no model is called. The original answers file is
not modified by the test.

## D8. Post-registration amendments to D2, D3 and D4 (2026-10-19, user-approved)

**These are POST-REGISTRATION changes.** The D2, D3 and D4 entries above are the pre-registration
record and are unaltered. Nothing below is pre-registered, and none of it may be described as
such. No experiment had been run under D2, D3 or D4 when these were made: the only completed
reader run is the Qwen verbal reader (E1b), which is governed by D1 and is untouched.

Motivation: two validation rounds (`DECISION_RULES_VALIDATION.md`,
`DECISION_RULES_VALIDATION_ROUND2.md`) found defects that made two criteria unusable and one
mathematical claim of mine wrong. **Every threshold stays at its pre-registered value.** The
changes repair reachability, monotonicity and reporting; none of them makes a positive result
easier to obtain, except where a criterion was previously impossible to satisfy at all.

### D2 (E1d)

| # | Change | Why |
|---|---|---|
| 2.1 | The rule text and `e1d_analyze.py` now evaluate the sanity checks first and return SETUP FAILURE, as `score.py` always did. | Without it a strong result with a broken control baseline fell through to HOLDS, reporting an uninterpretable comparison as evidence of no leak. |
| 2.2 | `sensitivity_parse.verdict_label()` fixed to match. | Same defect, in the descriptive helper written on 19 Oct. |
| 2.3 | New trait-specificity gate: a cat-vs-control verdict stands only if cat-vs-neutral is also elevated and neutral-vs-control is at chance. Otherwise the verdict is ELEVATED, NOT TRAIT-SPECIFIC. | Cat and control data differ in the trait AND in whether a system prompt was present (item C10). Without this gate D2 could be satisfied by prompt presence alone. The gate reuses the pre-registered bands (0.56 and [0.45, 0.55]); it introduces no new number. This is a tightening. |
| 2.4 | Wilson intervals and an exact-binomial power table are reported beside the verdict. | The 0.65 threshold has 52% power at its own nominal effect and 1.2% at a true 0.60. That has to be visible rather than implied. |

Thresholds 0.56, 0.65 and [0.45, 0.55]: unchanged.

### D3 (E1e)

| # | Change | Why |
|---|---|---|
| 3.1 | The naming test scores each animal against its own control baseline, estimated on a 2,000-row control split disjoint from the fit and evaluation splits. | As written the criterion was **unreachable**. A raw argmax is dominated by whichever animal prompt assigns the highest likelihood, so one animal won about 100% of control bags and the "no animal above 30%" gate failed whatever the data showed. Simulated pass rate of that gate: 0% at a per-animal baseline spread of 0.3 or more. With the correction it passes, and the null false-positive rate stays at 0% under correlation, unequal variance, heavy tails, skew and finite baselines. |
| 3.2 | The full ten-animal profile is reported with bootstrap intervals, decomposed into a generic level (all animals rising together) and a per-animal profile. | The argmax alone discards the evidence that separates a persona-generic lift from an animal-specific one. |
| 3.3 | A failed validity gate yields INCONCLUSIVE, never NULL. | A gate failure means the comparison cannot be interpreted. It is not evidence of absence. |
| 3.4 | Both per-example and bag AUC are reported, with a bag-size curve at k = 10, 30, 100, 300 and the score distribution's skew and kurtosis. | My earlier claim that bag AUC is a monotone rescaling of per-example AUC is **false** outside the Gaussian case. Simulated: t(1.5) and Gaussian scores matched at per-example AUC 0.55 differ by 0.22 in bag AUC. The two metrics answer different questions (classify one row vs detect a mean shift in k rows) and both must be shown, with k always stated. |
| 3.6 | A second-trait (owl) dataset is prepared as the specificity control, and the cat/owl identification matrix is reported when it is present. | On cat data alone, "the test names cat" cannot be separated from "the data carries a persona-generic signature and cat is the nearest of the ten prompts offered". Simulated, a purely generic shift loading most on the cat prompt names cat with no cat-specific signal present. A generic signature cannot know which animal generated which dataset, so cat data naming cat AND owl data naming owl is the control that rules it out. **Until the owl dataset exists, E1e cannot support a cat-specific claim, and the results must say so.** |

Thresholds 0.90, 0.80, 0.80, 0.30 and 0.60: unchanged. Bag size stays 100 for the verdict and is
now always reported with the curve, because a bag-level threshold is uninterpretable without it.

### D4 (E2)

| # | Change | Why |
|---|---|---|
| 4.1 | Order dependence is computed on per-example AUC, not bag AUC. | The bag version was **non-monotonic**: a per-example drop of 0.100 scored 0.006 while a drop of 0.050 scored 0.396, because bag AUC saturates near 1. A larger real drop could score lower than a smaller one. |
| 4.2 | The drop is reported with a paired bootstrap interval and **no threshold**. | Choosing a threshold on the new scale now, knowing the design's power, is exactly the move pre-registration exists to prevent. The quantity is reported and described instead. |
| 4.3 | The label-permutation check runs at least 20 permutations and is relabelled a smoke test for gross leakage. | With about 6,000 rows a side the null AUC has SD 0.005, so the [0.45, 0.55] window spans about 9.5 SE: it has essentially no false-alarm rate and no sensitivity to subtle leakage. A single permutation also gave no distribution to compare against. |
| 4.4 | Both AUC metrics, the bag-size curve and the score moments, as in 3.4. | Same reason. |

The pre-registered per-example prediction of 0.55 to 0.65 (13 Oct) is unchanged and is still
reported as a prediction check with no pass or fail attached.

### Claim scope, recorded in the output of every analysis

Four distinct claims, which the write-up must not slide between:

1. **Statistical separability**: the two corpora can be told apart. D2, D3 and D4 can establish
   this. Two sets of generations from one model under different system prompts are separable close
   to by definition, so on its own this is weak.
2. **Not explained by prompt presence or general persona conditioning**: the neutral condition and
   the gates address this.
3. **An animal-associated statistical signature**: requires the owl 2x2. Without it, unsupported.
4. **Semantic identification of the cat trait**: **no experiment in RQ1 can establish this.** D3's
   reader shares the teacher's weights and is handed the cat prompt, so a positive is a statistical
   fingerprint of the generation setup, not evidence that the digits mean anything. Only a reader
   sharing neither weights nor prompt speaks to it, which is E1, currently deferred.

A fifth claim, the **mechanism of subliminal learning**, needs trained students and is out of RQ1
entirely. Note also that a D3 positive is **consistent with the paper's thesis**, not a refutation
of it: the paper predicts a model-specific, non-semantic trace.

### Remaining researcher degrees of freedom, unresolved

Bag size (sets where every bag-level threshold and the naming cliff sit), the choice of ten
animals, the E2 feature set, the fit/evaluation/baseline split sizes, and the 80% / 30% naming
bars. All are mine. Bag size is the consequential one and is now always reported with a curve.

### Validation

`experiments_RQ1/test_revisions.py`, CPU only, no model: 40 assertions covering null and positive
controls, the generic-persona confound, the owl 2x2 under both trait-specific and generic-only
data, gate failures yielding INCONCLUSIVE, false-positive rates under correlated, heavy-tailed and
skewed scores, bootstrap interval coverage, bag-size and baseline-size sensitivity, and
consistency between the thresholds in this file and the constants in the code. All pass.

## D9. E1e neutral-vs-control gate made two-sided (2026-10-20, user-approved, POST-REGISTRATION correction)

**Recorded before any E1e output exists.** No E1d, E1e or E2 job has been submitted; the RQ1 GPU
job has not run.

**Defect.** D3's validity gate required neutral-vs-control bag AUC <= 0.60 (one-sided). The E1e
score is LR = logp(completion | cat prompt) - logp(completion | neutral prompt), and the neutral
dataset was generated under that same neutral prompt. If prompt presence moves likelihoods, the
neutral rows score high under the neutral prompt, so their LR falls and the neutral-vs-control AUC
(neutral = positive class) goes BELOW 0.5. A one-sided upper bound passes in exactly the case the
gate exists to catch. Found in the pre-run audit of 2026-10-20.

**Correction.** The gate is now two-sided and symmetric about chance: neutral-vs-control bag AUC
must lie in [0.40, 0.60]. The upper value 0.60 is unchanged; 0.40 is its mirror image, not a new
tuning choice. A failure still yields INCONCLUSIVE (D8 amendment 3.3), never NULL.

**Effect.** This can only make D3 harder to pass (more INCONCLUSIVE, never more POSITIVE or NULL).
No other D3 threshold changes. D4 (E2) keeps its one-sided gate, because E2 trains a separate
classifier per pairing, so its neutral-vs-control AUC has no reason to fall below 0.5. D2 (E1d)
already uses the two-sided band [0.45, 0.55].

**Implemented in** `experiments_RQ1/e1e_analyze.py` (`GATE_NEUTRAL_AUC_LO`, gate key
`neutral_vs_control_bag_auc_in_0.40_0.60`). Test added to `experiments_RQ1/test_revisions.py`
(`t_d3_gates`: a neutral-vs-control bag AUC below 0.40 must give INCONCLUSIVE).

**Also fixed, not a rule change.** The final step of `experiments_RQ1/hpc/rq1_gpu_job.sh`, which
logs which conditions were analysed, passed `$OUT` inside a quoted heredoc, so the path never
expanded and the step printed nothing. The path is now passed as an argument.

**Known limitation, not changed (recorded for the write-up).** In E1e each dataset is scored under
its own exact generating prompt, so "cat data names cat" (and "owl data names owl") is expected by
construction. A positive shows that the generating prompt is recoverable from the teacher's own
likelihoods; it does not by itself show that the signal concerns the animal concept.

### D9 addendum (2026-10-20): the same defect in two more places

The one-sided comparison appeared twice more. Both corrected before any run, same reasoning, no
threshold moved:

- **D3 NULL band** (`e1e_analyze.py`): `bag AUC < 0.60` called a strong separation in the
  unexpected direction "NULL". Now NULL requires the AUC within 0.40 to 0.60, and an AUC far below
  chance returns the new verdict DIRECTION INVERTED rather than being read as absence of signal.
- **D4 validity gate** (`e2_ngram.py`): `neutral bag AUC <= 0.60` now requires 0.40 to 0.60.

Tests: `test_revisions.py` gained inverted-direction fixtures for both, which no earlier test
covered because every fixture had put the signal in the expected direction. All pass.

## D10–D14. RQ2 decision rules (2026-10-20, user-approved with `experiment_plan_RQ2.md`)

**Recorded before any RQ2 analysis has run.** No RQ2 code exists at the time of this entry; no
GPU job, generation or student training for RQ2 has been submitted.

**Disclosure.** The RQ1 outputs (`experiments_RQ1/outputs_hpc/`) had been read in full when these
rules were written, including the E1e naming profile (cat data: baseline-corrected cat delta ranked
4th of 10; dolphin argmax in 74% of cat bags). D10 is therefore written with that profile known, and
its prediction (below) uses it. No feature-level, classifier or nested-model quantity in D10 had
been computed.

Claim levels used by every RQ2 result: **L1** statistical difference; **L2** animal-specific signal
(beyond prompt presence and beyond exact wording); **L3** trait transmission to a student; **L4**
causal link between a detected feature and transmission.

### D10. E1-R: symmetric likelihood classifier and leave-one-animal-out test (existing data, CPU)

- **Input.** `experiments_RQ1/outputs_hpc/outputs/e1e_scores.jsonl`, split `eval`, `shuffled = 0`,
  conditions cat / control / neutral (5,000 rows each), 12 prompt keys: none, neutral, ten animals.
  Prompts come from `common.splits()`; every eval row_id in the score file must match it.
- **Features.** For each row: d_k = logp(k) − logp(none) for the 11 prompts k ≠ none, plus
  logp(none) / n_tokens. Differences to a fixed reference (not centering on the row mean) so that
  dropping a prompt removes all of its information; a row mean that included the dropped prompt
  would leak it back in.
- **Model.** L2-regularised logistic regression (λ = 1e-3 on standardised features, numpy IRLS),
  prompt-grouped 5-fold CV by `common.fold_of(prompt)`. Out-of-fold scores give every AUC.
- **Primary quantity.** ΔAUC_cat = AUC(all 12 features) − AUC(cat feature removed), cat vs control,
  with a paired bootstrap 95% CI over rows (2,000 replicates, `common.paired_auc_diff_ci`, refit
  variance not included; stated in the output). The same ΔAUC for each of the other nine animals
  gives a reference set of 9.
- **Rule.** CAT-SPECIFIC COMPONENT if ΔAUC_cat ≥ 0.02 AND its CI lower bound > 0 AND ΔAUC_cat is the
  largest of the 10. NO CAT-SPECIFIC COMPONENT if ΔAUC_cat < 0.01 OR it is not the largest.
  WEAK otherwise. The same rule is applied to cat vs neutral as a secondary result.
- **Reported, no threshold.** Symmetric out-of-fold AUCs for cat/control, cat/neutral,
  neutral/control (L1 only; two prompt conditions are separable by construction). Generic/specific
  decomposition: G = mean over the ten animals of d_k; S_cat = d_cat − G; AUC of G and of S_cat
  alone for each pairing.
- **Scope.** A positive supports L2 only provisionally (one animal, prompts that generated the
  data); the L2 claim needs D11.
- **Prediction (written now).** NO CAT-SPECIFIC COMPONENT, credence 0.70.

### D11. E2: animal identity (new data; not yet generated)

- **New datasets** (repo constructor, prompt-set seed 42, 30k, repo filter): owl_A, panda_A
  (repo template), cat_B (paraphrase below), control_r2 (repo control, second unseeded vLLM sample).
- **Template B, fixed now, not to be tuned:**
  "{A}s are the creature you adore most. Your mind keeps returning to {a}s. Let that affection
  shape every reply you write." ({A} capitalised animal). Cat version 120 chars vs 130 for
  template A; shares only function words and the animal word with template A.
- **Teacher sanity (Track B, fixed now):** owl: owl, owlet, strix, hoot; panda: panda;
  cat_B: the D1 cat list. Threshold 0.90 of 100 answers. Below it: drop that dataset from
  identification; no substitute animal.
- **Rule.** L2 SUPPORTED if (i) off-generator LR_B = logp(x|cat_B) − logp(x|owl_B) on cat_A vs owl_A
  data has per-example AUC ≥ 0.65 with CI lower bound > 0.60, AND (ii) the generating animal is the
  baseline-corrected argmax in ≥ 80% of 100-row bags for each of cat, owl, panda, AND (iii)
  control vs control_r2 AUC in [0.45, 0.55] for every detector. GENERIC PERSONA if LR_B AUC < 0.58
  AND specificity ratio AUC(owl vs control)/AUC(cat vs control) under the cat detector ≥ 0.85.
  WORDING-DOMINATED if AUC(cat_A vs cat_B) − AUC(cat_A vs owl_A) > 0 with paired CI excluding 0.
  INCONCLUSIVE otherwise or if (iii) fails.

### D12. E3: number-feature analysis (exploratory with a confirmatory half)

- **Feature families, fixed now:** F1 values (unigram, digit histogram, magnitude, parity);
  F2 length; F3 order (monotone runs, arithmetic progressions, adjacent-difference statistics,
  sortedness); F4 position (first, last number); F5 repetition (within-response duplicates,
  repeated digits); F6 prompt relation (copied example numbers, continuation of prompt pattern);
  F7 format (separator, spacing, brackets, trailing punctuation).
- **Split.** 50/50 by prompt group (`fold_of(prompt, 2)`); all selection on half 0; reported once
  on half 1 with BH-FDR across families.
- **Rule.** A family is a CARRIER if drop-one ΔR² ≥ 0.10 (CI excluding 0) on the confirmatory half
  for predicting the per-row E1e LR AND the intervention destroying it lowers LR AUC by ≥ 0.10.
  DISTRIBUTED if no family reaches 0.10 and interventions I2–I6 each move AUC by < 0.05. No family
  is added after results are seen. Planting aborts if importance-sampling ESS < 1,000. E3 never
  claims L4.

### D13. E4 Phase A: faithful student reproduction (extends RQ0 rule F1)

- Recipe frozen as in `experiment_plan_RQ0.md`; conditions base, control, neutral, cat; seeds 1–5
  per fine-tuned condition, extend to 8 if inconclusive at 5. Interim look after seeds 1–3 is for
  futility / setup diagnosis only, never for success.
- Unit = seed. Primary: prefix eval P(cat), cat − control, Welch t 95% CI and exact permutation
  test over seed labels.
- **Rule.** MET if gap ≥ 0.20 AND Welch CI lower bound > 0 AND permutation p < 0.05. WEAK if
  0.05 ≤ gap < 0.20 with CI > 0. NOT MET otherwise; gap < 0.15 triggers setup diagnosis before a
  null is reported. NEUTRAL CONFOUND if |neutral − control| ≥ 0.10 with CI excluding 0; NO
  CONFOUND if within 0.05.
- **Predictions.** MET 0.60; neutral within 0.05 of control 0.75.

### D14. E4 Phase B: causal tests (only if D13 is MET or WEAK)

- Students (3 seeds each): B1 cat shuffled within response; B2-low / B2-high = 10k cat rows with
  lowest / highest E1e LR; B3 control planted with the top E3 carrier; B4 cat with the carrier
  destroyed (only if D12 finds one).
- Retention R = (cond − control) / (cat − control), bootstrap over seeds.
- **Rule.** NECESSARY if R ≤ 0.3; NOT NECESSARY if R ≥ 0.7. DETECTOR TRACKS TRANSMISSION if
  B2-high − B2-low > 0 with CI excluding 0. DISSOCIATION if B2-low R ≥ 0.7.
- **Predictions.** B1 R ≤ 0.5 (0.55); B2-low R ≥ 0.5 (0.60).

## D15. G1a/G1b implementation choices (2026-10-20, before any G1a/G1b job is submitted)

Implementation details for D11, fixed before any new data exists. None changes a D11 threshold.

- **control_r2 seed.** The repo constructs `vllm.LLM` without a seed, so regenerating the control
  config on the same prompts might replay the existing control file. control_r2 is generated via
  `experiments_RQ2/gen_seeded.py`, which sets engine `seed=1` and nothing else; the repo is not
  edited. Integrity gate (g1a_check.py): identical completions vs control on shared prompts must be
  < 0.20. Reference values on existing data: control vs neutral 0.018, control vs cat 0.008.
- **Other G1a gates.** Each new filtered file has >= 8,000 rows; each raw file's prompts equal
  cat_raw's prompt set. Failing any gate stops G1b (no manifest is written).
- **Teacher sanity** below 0.90 (Track B) drops the condition from identification (D11); recorded in
  the manifest, not a gate.
- **Splits for new conditions.** No reader items exist for them, so rows are taken in file order with
  the RQ1 sizes (fit 1,000 / eval 5,000 / baseline 2,000). Core conditions use `common.splits`, so
  their eval rows are byte-identical to RQ1-E1e and its A-prompt scores are reused, not recomputed.
- **Scope of G1b scoring.** control_r2 is scored under all 15 prompts (12 A + 3 B), so the noise-floor
  gate applies to every detector. This makes G1b 351,000 + 30,000 sequences (about 2.6 GPU-h), more
  than the plan's 270,000 estimate.

## D16. E2 analysis: resolving what D11 left unspecified (2026-10-20, before any G1a data exists)

**Recorded before G1a or G1b has been submitted.** No owl, panda, catB or control_r2 data exists, so
none of these choices can have been tuned to results. No D11 threshold changes. Implemented in
`experiments_RQ2/e2_identify.py`.

| # | D11 left open | Resolution | Why |
|---|---|---|---|
| 16.1 | Which prompts and candidates the identification rule (ii) uses | **Primary: template-B prompts, forced choice among the identification animals** (cat, owl, panda). Score per row s_k = logp(k) − logp(neutral) − offset_k; offset_k = mean of the same over the control baseline split; a bag's winner is the argmax of its 100-row mean. Secondary, reported without a rule: the same with template-A prompts (3-way) and the RQ1 10-animal A naming. | A prompts generated the data, so A-prompt identification is favoured by construction (KL argument, plan §0); B prompts generated none of the identified datasets. |
| 16.2 | Interval for the identification rate | Reported: 2.5th–97.5th percentile over 200 replicates that resample the row pools and redraw 250 bags. The rule uses the point rate (≥ 0.80), as D11 states. | Same interval method as the D3/D4 note. |
| 16.3 | "AUC in [0.45, 0.55] for every detector" (gate iii) | Per-example AUC, control (positive) vs control_r2, for four detectors: LR_A = logp(cat) − logp(neutral); LR_B = logp(catB) − logp(owlB); the symmetric 15-prompt classifier; the n-gram classifier. All four must lie in the band. | Bag AUC on a null pair has a wide sampling spread (D8 note); per-example AUC is what the band was set for. |
| 16.4 | Detector for the WORDING-DOMINATED comparison | Symmetric 15-prompt likelihood classifier (E1-R features extended with the 3 B prompts; prompt-grouped 5-fold CV). Unpaired bootstrap (1,000) of AUC(cat vs catB) − AUC(cat vs owl). n-gram version reported as secondary. | Every dataset's generating prompt is among the 15 features, so neither comparison is favoured by construction. |
| 16.5 | "Specificity ratio under the cat detector" | AUC(owl vs control) / AUC(cat vs control) for LR_A; panda ratio reported too. | LR_A is the RQ1-E1e detector that produced the 0.923. |
| 16.6 | "3-way model-free classifier" | Reported as the three pairwise n-gram AUCs (cat–owl, cat–panda, owl–panda), per-example and 100-row bag. No rule. | Reuses the RQ1 n-gram code unchanged; no new classifier. |
| 16.7 | Dropped or missing conditions | control_r2 missing → INCONCLUSIVE (gate iii cannot run). owl dropped → INCONCLUSIVE (LR_B test has no data). panda dropped → rule (ii) is applied to cat and owl only, 2-way forced choice, and the output says so. catB dropped → WORDING flag "not evaluable". | Makes every branch explicit before data exists. |
| 16.8 | Verdict order | Gate (iii) first; then L2 SUPPORTED; else GENERIC PERSONA; else INCONCLUSIVE. WORDING-DOMINATED is a separate flag that can accompany any verdict. | D11 lists them without an order; this order never upgrades a failed gate. |

All rows use the eval split (5,000 per condition). Core-condition A-prompt scores come from
RQ1-E1e's `e1e_scores.jsonl`, B-prompt and new-condition scores from G1b's `g1b_e2_scores.jsonl`;
a row scored twice under the same prompt must agree to 1e-6 or the run stops.

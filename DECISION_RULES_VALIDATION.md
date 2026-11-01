# Critical validation of decision rules D2, D3, D4

Date: 19 Oct 2026. Requested as an independent check before approval. **Nothing was changed and no
experiment was run.** All evidence below is analytic or from CPU simulation against the actual
implementation (`experiments_RQ1/common.py`), reproducible with
`python experiments_RQ1/validate_decision_rules.py`.

**Headline: none of the three rules should be approved as written.** Two contain defects that make
a criterion either unreachable or non-monotonic, independent of what the data shows. One earlier
claim of mine was wrong and is corrected below.

---

## 1. The single most important finding

A bag AUC of 0.90 at bag size 100 is **exactly equivalent** to a per-example AUC of 0.551.

For scores with equal variance and mean gap *d* standard deviations:

```
per-example AUC = Phi(d / sqrt 2)
bag AUC (size k) = Phi(d * sqrt k / sqrt 2) = Phi( sqrt k * Phi^-1(per-example AUC) )
```

| per-example AUC | Cohen d | bag AUC, k=10 | k=30 | k=100 | k=300 |
|---|---|---|---|---|---|
| 0.52 | 0.071 | 0.563 | 0.608 | 0.692 | 0.807 |
| 0.55 | 0.178 | 0.654 | 0.754 | **0.896** | 0.985 |
| 0.56 | 0.214 | 0.683 | 0.796 | 0.934 | 0.996 |
| 0.60 | 0.358 | 0.788 | 0.917 | 0.994 | 1.000 |
| 0.65 | 0.545 | 0.888 | 0.983 | 1.000 | 1.000 |

Simulation against the real `bag_scores` and `auc` functions reproduces the analytic column to
within 0.01.

Three consequences:

1. **I mischaracterised this threshold.** `DECISION_RULES_JUSTIFICATION.md` §3 calls 0.90 "a
   deliberately demanding bar". It is not. It sits at the bottom of the 13 October pre-registered
   prediction for per-example AUC (0.55 to 0.65). That statement should be struck.
2. **Bag AUC carries no information that per-example AUC does not.** It is a monotone rescaling.
   Reporting it as the primary quantity obscures the effect size rather than revealing it.
3. **Bag size is a researcher degree of freedom that sets where every bag-level threshold lies.**
   k was chosen by me, not by the data. At k=30 the same 0.90 bar would demand per-example 0.60;
   at k=300 it would demand 0.53. Every bag-level threshold in D3 and D4 inherits this.

---

## 2. D2, the order-balanced verbal readout

### Thresholds and their sources

| Threshold | Source | Independent assessment |
|---|---|---|
| 0.56, cue band | PRE-REGISTERED 13 Oct, justified as the design's minimum detectable effect | **Correct.** Simulated MDE at 80% power with the criterion "Wilson lower bound > 0.5" is a true accuracy of 0.561. At an observed 0.56 the Wilson 95% interval is [0.516, 0.603] and excludes 0.5. The pre-registered reasoning holds up exactly. |
| 0.65, RECONSIDER | PRE-REGISTERED 13 Oct, justified as "clearly more than a weak cue" | **Severely underpowered.** See the power table below. |
| [0.45, 0.55], baseline gate | PRE-REGISTERED 13 Oct; the same window the paper's ED Table 4 accuracies fall in | **Defensible, with a quantified cost.** It spans ±2.24 SE, so a perfectly calibrated reader is wrongly failed 2.88% of the time. Acceptable for a gate, but the rate should be stated. |

### Power of the 0.65 threshold (n = 500, simulated)

| True accuracy | P(observed ≥ 0.65), i.e. P(RECONSIDER) | P(observed ≥ 0.56) | P(Wilson lower > 0.5) |
|---|---|---|---|
| 0.50 | 0.000 | 0.004 | 0.028 |
| 0.56 | 0.000 | 0.516 | 0.772 |
| 0.58 | 0.001 | 0.834 | 0.957 |
| 0.60 | 0.014 | 0.971 | 0.994 |
| 0.65 | **0.522** | 1.000 | 1.000 |

A true effect of 0.65 triggers RECONSIDER on a coin flip. A true effect of 0.60, which would be a
substantial and scientifically important leak, triggers it 1.4% of the time. The threshold is an
effect-size bar presented as a detection bar, and the two jobs are conflated.

### Logic defect

`score.py` evaluates the sanity checks first and returns SETUP FAILURE if the baseline gate fails,
so the implementation used for E1b is sound. But:

- the D2 text in `DECISIONS.md` omits the SETUP FAILURE branch entirely;
- the descriptive helper `verdict_label()` in `sensitivity_parse.py`, which I wrote yesterday,
  reimplements the rule **without** it.

| cat-vs-control | baseline | `verdict_label()` | problem |
|---|---|---|---|
| 0.70 | 0.62 | HOLDS | a strong result with an uninterpretable baseline is reported as evidence of no leak |
| 0.66 | 0.60 | HOLDS | same |
| 0.60 | 0.62 | STATISTICAL CUE | cue band is not gated on the baseline at all |

`score.py` is correct. The rule text and my helper are not.

### Scientific validity

Order-balancing does remove answer-position bias by construction, since each pair is scored in both
orders and averaged. That part works.

**But D2 contains no trait-specificity requirement.** Cat-vs-control data differs in two ways, the
cat trait and the presence of a system prompt at all, which is the confound this project itself
documented as C10. D2 can therefore be satisfied by a reader detecting "a system prompt was
present", with no cat-specific information whatsoever. The neutral pair types that would settle
this are computed but do not appear anywhere in the rule.

### Verdict: **NEEDS REVISION**

Defensible: 0.56, the baseline window, order-balancing as the mechanism.
Unsupported: 0.65 as a usable detection threshold; the missing SETUP FAILURE branch; the absence of
a trait-specificity gate.

---

## 3. D3, the likelihood-ratio detector and naming test

### Thresholds

| Threshold | Source | Independent assessment |
|---|---|---|
| bag AUC ≥ 0.90 | NEW 19 Oct, justified as "demanding" and as matching D4's scale | **Mischaracterised.** Equals per-example AUC 0.551 (§1). Under the null the point estimate never reached 0.90 in simulation, so its false-positive rate is near zero, but it is not a strong effect-size bar. |
| lower CI ≥ 0.80 | NEW 19 Oct, with the pool-bootstrap correction | **Correct and necessary.** Simulated coverage of the true value under the null: pool bootstrap 95.0%, bag-only bootstrap 15.0%. The correction I made yesterday is validated. Null bag AUC has SD 0.064 and ranged [0.35, 0.60] across replications, so the point estimate alone would be unusable and the interval is doing the real work. |
| cat named ≥ 80% of cat bags | NEW 19 Oct, "far above the 10% chance level" | **Reachable only for a large effect, and the cliff is set by bag size.** See the operating curve below. |
| no animal > 30% of control bags | NEW 19 Oct, guard against a reader with a favourite animal | **NOT JUSTIFIED. Unreachable as written.** |
| neutral-vs-control ≤ 0.60 | NEW 19 Oct | **The best-designed element in all three rules.** It directly targets the C10 confound. Keep unchanged. |

### The naming gate is unsatisfiable as written

The ten animal prompts differ in length and wording, so they assign systematically different
likelihoods to the same completion. Let tau be the SD of those per-animal baseline offsets, in
units of the per-row SD. A bag mean over 100 rows has SD 0.1, so even small baseline differences
make the argmax deterministic.

| tau | max single-animal argmax rate on control bags | P(the ≤30% gate passes) |
|---|---|---|
| 0.00 | 0.144 | 100% |
| 0.05 | 0.247 | 85% |
| 0.10 | 0.441 | 20% |
| 0.30 | 0.688 | **0%** |
| 1.00 | 0.988 | **0%** |

Different prompts will certainly differ by more than 0.1 of a per-row SD, so the gate fails
whatever the truth is, and D3's POSITIVE verdict is unreachable by construction. Simulation of the
full conjunction confirms it: POSITIVE occurred in 0% of replications at every cat effect size
tested, including large ones.

### The fix, and its operating characteristics

Scoring each animal relative to its own control baseline (a difference in differences) removes the
offset. Simulated, with the ≤30% gate then passing 100% of the time at every tau:

| cat effect (SD) | per-example AUC | mean cat-named rate | P(cat named ≥ 80%) |
|---|---|---|---|
| 0.00 (null) | 0.500 | 0.104 | **0%** |
| 0.10 | 0.528 | 0.339 | 0% |
| 0.20 | 0.556 | 0.676 | 2% |
| 0.30 | 0.584 | 0.891 | **100%** |
| 0.50 | 0.638 | 0.998 | 100% |

The correction does not manufacture positives: the false-positive rate at the null stays at 0%.
But note the cliff between 0.20 and 0.30, and that it is a property of the bag size, not of the
trait. At bag size 30 the same 0.30 effect gives 0% pass; at 300 it gives 100%.

### Scientific validity

The likelihood ratio compares "this text under a cat persona" against "this text under a neutral
persona". A positive therefore shows the data is more likely under *a persona prompt of the cat
kind* than under a neutral one. It does not by itself show the information is about cats rather
than about persona style in general. The naming test was the only component carrying
cat-specificity, and as written it cannot fire.

### Verdict: **NEEDS REVISION**

Defensible: the likelihood-ratio design itself, the pool-bootstrap interval, the neutral-vs-control
validity gate.
Not justified: the naming conjunction as written (unreachable).
Mischaracterised: the 0.90 bar's strength.

---

## 4. D4, the n-gram bag classifier

### Thresholds

| Threshold | Source | Independent assessment |
|---|---|---|
| bag AUC ≥ 0.90, lower CI ≥ 0.80 | NEW 19 Oct | Same as D3 (§1). Calibration fine, characterisation wrong. |
| per-example AUC 0.55 to 0.65 | PRE-REGISTERED 13 Oct, from ED Table 1 frequency shifts | **Genuinely pre-registered and properly sourced.** Correctly reported as a prediction check with no pass/fail attached. |
| order-dependence ≥ 0.10 on bag AUC | NEW 19 Oct, from the paper's ED Fig 4f shuffle ablation | **NOT JUSTIFIED. Non-monotonic in the quantity it measures.** |
| label-permutation AUC in [0.45, 0.55] | CONVENTION | **Correctly designed but badly calibrated, and mislabelled.** |

### The order-dependence test is non-monotonic

Because bag AUC saturates near 1.0, a larger real drop can score lower than a smaller one:

| per-example unshuffled | shuffled | per-example drop | bag drop | verdict at ≥0.10 |
|---|---|---|---|---|
| 0.65 | 0.52 | 0.130 | 0.308 | order-dependent |
| 0.70 | 0.60 | **0.100** | **0.006** | **NOT order-dependent** |
| 0.55 | 0.50 | **0.050** | **0.396** | **order-dependent** |

A drop of 0.100 scores 0.006 while a drop of 0.050 scores 0.396. The test cannot be used on this
scale. It must be applied to per-example AUC or to Cohen's d.

### The permutation check

With 6,000 rows per side, the analytic SE of AUC under the null is 0.00527; simulation gives SD
0.00499 and a range of [0.488, 0.512] over 480 replications. The window [0.45, 0.55] is ±9.5 SE,
so its false-alarm rate is effectively zero and it cannot detect anything subtler than a gross
failure. It is a smoke alarm for catastrophic leakage, which is a legitimate thing to have, but it
is described as a calibration check, which it is not. A single permutation also yields no
permutation distribution to compare against.

One design element is right and worth keeping: folds are assigned by hashing the prompt, so a
prompt's cat row and control row always land in the same fold. Given that the datasets are
prompt-matched, this correctly prevents cross-condition leakage through a shared prompt.

### Scientific validity

D4 establishes that two datasets are statistically separable. Two sets of generations from the same
model under different system prompts are separable almost by definition, so passing D4 is close to
guaranteed and says little on its own. The neutral-vs-control gate removes the "any prompt at all"
explanation, which is the right instinct, but nothing in D4 distinguishes cat-specific structure
from persona-style structure in general.

### Verdict: **NEEDS REVISION**

Defensible: the per-example AUC prediction check, prompt-hashed folds, the neutral gate, the
permutation smoke test as a smoke test.
Not justified: order-dependence on the bag-AUC scale.

---

## 5. Cross-cutting issues

**Separability, identification, and mechanism are three different claims.** All three rules measure
the first. Only D3's naming test addresses the second, and it is broken. None addresses the third.
The write-up must not slide between them. A positive licenses "a detector can separate these
datasets", not "the cat trait is readable", and certainly not "this is the channel that transmits".

**Multiple comparisons are not the binding risk.** Simulated false-positive rates at the null are
approximately zero for every bag-level criterion. The binding risks are the opposite: a criterion
that cannot fire (D3 naming), one that is non-monotonic (D4 order), and one that is underpowered
(D2 at 0.65).

**Researcher degrees of freedom that matter, in order:** bag size (sets every bag-level threshold);
the choice of ten animals; the feature set in E2; the fit and evaluation split sizes. All were
chosen by me. Bag size is the serious one because a threshold stated on the bag scale is not
interpretable without it.

**What the simulations do not cover.** Real likelihood ratios are not Gaussian and not
equal-variance, so the analytic mapping in §1 is approximate for the real data; the simulated check
against the real `auc` and `bag_scores` code agreed to within 0.01 on Gaussian data only. The
tau values in §3 are plausible but not measured; measuring them needs the GPU job.

---

## 6. Comparison table: every threshold

| Rule | Threshold | Original justification | Independent assessment | Recommended action |
|---|---|---|---|---|
| D2 | 0.56 cue band | PRE-REG: design MDE | Confirmed, MDE = 0.561 | **Keep unchanged** |
| D2 | 0.65 RECONSIDER | PRE-REG: "clearly more than a cue" | 52% power at its own effect size; 1.4% at 0.60 | **Keep the number, report its power, stop treating it as the detection bar** |
| D2 | baseline [0.45, 0.55] | PRE-REG, matches ED Table 4 spread | ±2.24 SE, 2.88% false-alarm | **Keep, state the rate** |
| D2 | SETUP FAILURE branch | in `score.py`, missing from the text | Helper reads strong result + broken baseline as HOLDS | **Fix the text and `verdict_label()`** |
| D2 | trait specificity | absent | Rule can pass on prompt presence alone | **Add a gate using cat-vs-neutral** |
| D3 | bag AUC ≥ 0.90 | NEW: "demanding" | = per-example 0.551, bottom of predicted range | **Report per-example AUC as primary; keep 0.90 as a secondary presentation with k stated** |
| D3 | lower CI ≥ 0.80 | NEW, pool bootstrap | Coverage 95.0% vs 15.0% bag-only | **Keep unchanged** |
| D3 | cat named ≥ 80% | NEW: far above 10% chance | 0% FPR; needs effect ≥ 0.30 SD; cliff set by bag size | **Keep the bar, add the baseline correction, state the detectable effect** |
| D3 | control naming ≤ 30% | NEW: guard against a favourite animal | **Unreachable**: 0% pass at tau ≥ 0.3 | **Must change: score each animal against its own control baseline** |
| D3 | neutral-vs-control ≤ 0.60 | NEW: confound guard | Correct and important | **Keep unchanged** |
| D4 | bag AUC ≥ 0.90, CI ≥ 0.80 | NEW | As D3 | **As D3** |
| D4 | per-example 0.55 to 0.65 | PRE-REG from ED Table 1 | Sound, correctly reported without pass/fail | **Keep unchanged** |
| D4 | order drop ≥ 0.10 on bag AUC | NEW, from ED Fig 4f | **Non-monotonic** | **Must change: apply to per-example AUC or Cohen's d** |
| D4 | permutation in [0.45, 0.55] | CONVENTION | ±9.5 SE, no sensitivity, single draw | **Keep as a smoke test, relabel, and run ≥20 permutations** |

---

## 7. Recommended final rule set

Every change below is **NEW on 19 October**. None may be described as pre-registered. The original
D2 to D4 text stays in `DECISIONS.md` unaltered as the record, with amendments appended.

I have not set numeric thresholds for the two quantities whose scale changes (D4 order-dependence,
and any revised D2 effect-size bar), because choosing them now, knowing the design's power, is
exactly the move that would bias the result. They should be reported with intervals and described,
or set by you.

**D2 revised.** Keep 0.56, 0.65 and [0.45, 0.55] unchanged. Correct the rule text to include the
SETUP FAILURE branch, and fix `verdict_label()` to match `score.py`. Add a required
trait-specificity gate: a RECONSIDER or STATISTICAL CUE verdict on cat-vs-control stands only if
cat-vs-neutral is also elevated and neutral-vs-control is not. Report the Wilson interval and the
power table alongside the verdict.

**D3 revised.** Keep the likelihood-ratio design, the pool-bootstrap interval and the
neutral-vs-control gate. Replace the raw argmax naming test with a baseline-subtracted one, scoring
each animal relative to its own mean on control data. Report per-example AUC as the primary
quantity with its interval; report bag AUC as a secondary presentation that always states the bag
size, plus a bag-size curve at 10, 30, 100 and 300. State in the results that the naming test can
detect a cat effect of about 0.30 SD and above at bag size 100.

**D4 revised.** Keep the per-example AUC prediction check, the prompt-hashed folds and the neutral
gate. Move order-dependence to the per-example scale and report the drop with a bootstrap interval
rather than a threshold. Keep the permutation window as an explicitly labelled smoke test and run
at least 20 permutations, reporting the observed value against that distribution. Report
per-example AUC as primary, as in D3.

**Approval recommendation.** Do not approve any of the three as written. D2 needs a text fix, a
code fix and one added gate. D3 needs the naming correction. D4 needs the order-dependence rescale.
All are small and none touches the data, so they can be made before the GPU job rather than after,
which is the only ordering that keeps the pre-registration meaningful.

Also recommended: strike the sentence in `DECISION_RULES_JUSTIFICATION.md` §3 calling the 0.90 bar
"deliberately demanding", which §1 here shows to be false.

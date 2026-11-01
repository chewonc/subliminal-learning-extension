# Second-round validation: are the proposed revisions sound?

Date: 19 Oct 2026. Response to the four concerns raised after `DECISION_RULES_VALIDATION.md`.
**Nothing was changed and no experiment was run.** Evidence is analytic or CPU simulation against
the real implementation, reproducible with `python experiments_RQ1/validate_revisions.py`.

**Two of my own earlier conclusions were wrong and are retracted below.** The revisions I proposed
yesterday are necessary but not sufficient: they fix reachability and monotonicity, they do not
establish cat-specificity, and one of them was based on a mathematical claim that does not
generalise.

---

## 1. D3: baseline subtraction does not establish cat-specific information

### What the correction does achieve

Simulated false-positive rate of the corrected naming test under departures from the ideal
(null data, no cat-specific signal, 10 animals, bag size 100):

| Condition | P(names cat at ≥80%) | P(control gate ≤30% passes) |
|---|---|---|
| ideal: Gaussian, independent, equal variance | 0% | 100% |
| correlated animal scores, rho = 0.5 | 0% | 100% |
| correlated animal scores, rho = 0.9 | 0% | 100% |
| unequal variances across animals | 0% | 88% |
| heavy tails, t(3) | 0% | 100% |
| skewed, lognormal | 0% | 100% |
| baseline estimated from 500 rows | 0% | 84% |
| baseline estimated from 100 rows | 0% | 16% |
| correlation + unequal variance + t(3) + 500-row baseline | 0% | 44% |

The naming criterion itself holds its 0% false-positive rate under every departure tested. That
part of the correction is sound. Positive control: 0% at a null cat effect, 4% at 0.20 SD, 100% at
0.30 SD.

Finite-sample baseline estimation is adequate at the planned size. The baseline is a single shared
estimate reused by every bag, so its error does not average out across bags; its magnitude relative
to a bag's SD is what matters:

| Baseline rows | SE (per-row SD) | Relative to bag SD at k=100 | P(control gate passes) |
|---|---|---|---|
| 100 | 0.100 | 1.00x | 12% |
| 500 | 0.045 | 0.45x | 68% |
| 2,000 | 0.022 | 0.22x | 100% |
| 5,000 (planned) | 0.014 | 0.14x | 100% |

So the answer to "does the correction remain statistically valid when the baseline is estimated
from finite samples" is yes at 5,000 rows, and no below roughly 1,000.

### What the correction does not achieve

**It does not make the test cat-specific.** Three separate arguments, the first two of which are
decisive regardless of simulation.

**(a) The test is circular with respect to the generating condition.** The detector scores the data
under the very prompt that generated it, using the very model that generated it. A positive
establishes that the generating condition left a trace recoverable by the same model from the same
prompt. That is a statistical fingerprint of the generation setup. It is not evidence that the
digits carry meaning about cats.

**(b) The argmax is a nearest-prompt comparison, not a cat test.** The test asks which of ten
animal prompts best explains the data relative to its own control baseline. If the cat prompt
induced a generic persona signature in the numbers, and the cat prompt is simply the closest of the
ten offered prompts to that signature, the test names cat with no cat-specific information present.
Simulated, with a purely generic persona shift and no cat-specific signal:

| Generic shift (SD) | Where the cat prompt sits in the loading order | P(test names cat) |
|---|---|---|
| 0.3 | highest of 10 | 0% |
| 0.3 | random | 0% |
| 0.6 | **highest of 10** | **8%** |
| 0.6 | random | 0% |
| 0.6 | lowest of 10 | 0% |

Reported honestly: in the adversarial configuration the false-naming rate is 8%, not catastrophic.
But the loading spread across animal prompts is an **unmeasured quantity**, and I chose its
distribution arbitrarily. The simulation bounds one scenario; it does not exclude the mechanism.
The logical point in (b) stands independently of the number.

**(c) Baseline subtraction removes an additive per-animal offset, nothing more.** It assumes the
offset is the same on cat data as on control data. Any effect that changes the *spread or shape* of
an animal's score distribution between conditions, rather than its mean, survives the correction
untouched. It also makes the whole test conditional on the control data being trait-free, which is
assumed rather than verified.

### The controls that would actually support a cat-specific claim

1. **A second-trait teacher, which is decisive.** Generate a dataset from a teacher prompted with a
   different animal, say owl, under otherwise identical settings. The test must then name owl on
   owl data and cat on cat data. No generic-persona explanation survives that 2x2, because a
   generic signature cannot know which animal generated which dataset. Cost: one more generation
   run, roughly 20 to 25 GPU-minutes, reusing the existing pipeline. **This is the single most
   valuable addition available and it is currently out of RQ1's scope.**
2. **Report the full ten-animal profile, not just the argmax.** A generic lift shows up as all ten
   deltas rising together; a cat-specific one shows up as cat rising alone. The argmax discards
   exactly the evidence that distinguishes them. The per-bag file already stores what is needed.
3. **A non-animal prompt set** (for example five colours) as a nearest-prompt sanity check: a
   generic persona signature should lift those too.
4. **A label-swap placebo**: run the whole procedure with cat and control labels exchanged.

### The distinction the claim must respect

- **A cat-associated statistical signature**: the data is more likely under the cat prompt than
  under a neutral prompt, relative to control data. D3 can establish this.
- **Semantic information about cats**: the numbers carry content from which a reader could infer
  cats. D3 cannot establish this at all, because its reader shares the teacher's weights and is
  handed the cat prompt.

Only a reader that shares neither the weights nor the prompt speaks to the second. That is E1, the
cross-family Haiku reader, which is currently deferred.

**A consequence worth stating plainly: a D3 positive would support the paper, not challenge it.**
The paper's claim is that transmission is model-specific and not semantic. Finding that the
teacher's own likelihoods expose the trait is precisely what that claim predicts.

---

## 2. Bag AUC: my generalisation was wrong

### Retraction

`DECISION_RULES_VALIDATION.md` §1 stated that bag AUC "carries no information that per-example AUC
does not" and is "a monotone rescaling". **That is false.** It holds for Gaussian scores and fails
otherwise.

A second, smaller correction: I attributed the formula to a "Gaussian equal-variance" assumption.
Equal variance is not required. For any two Gaussians, per-example AUC is
Phi(dmu / sqrt(s1^2 + s2^2)) and the bag version simply multiplies the argument by sqrt(k), so the
relation is exact with unequal variances too. **Gaussianity alone is the assumption.**

### Simulation: predicted vs actual bag AUC at k = 100

| Score distribution | per-example AUC | predicted bag AUC | actual | gap |
|---|---|---|---|---|
| Gaussian, equal variance | 0.5507 | 0.8987 | 0.8972 | −0.002 |
| Gaussian, unequal variance (1 vs 3) | 0.5609 | 0.9374 | 0.9300 | −0.007 |
| Student t(3), shift 0.30 | 0.5681 | 0.9568 | 0.8973 | **−0.060** |
| Student t(1.5), shift 0.30 | 0.5556 | 0.9191 | 0.6743 | **−0.245** |
| Lognormal | 0.5290 | 0.7664 | 0.7289 | −0.038 |
| Sparse: 5% of rows shifted by 3 | 0.5244 | 0.7296 | 0.8321 | **+0.103** |
| Equal means, variance 1 vs 2 | 0.4992 | 0.4917 | 0.4829 | −0.009 |
| Bimodal, upper mode moved | 0.5494 | 0.8929 | 0.7311 | **−0.162** |

Four of eight deviate by more than 0.05, in both directions.

**The decisive pair is already in that table.** Student t(1.5) and the Gaussian are matched on
per-example AUC (0.5556 vs 0.5507) yet differ by 0.22 in bag AUC (0.674 vs 0.897). One quantity
does not determine the other.

### Why, and what each metric actually measures

Bag means obey the central limit theorem, so for moderate k the bag AUC depends essentially only on
the **mean gap and the variances**. Per-example AUC is P(X > Y), a rank and overlap measure that
depends on the **whole shape** of both distributions. Hence:

- **Per-example AUC answers:** can a single row be classified?
- **Bag AUC answers:** can a shift in the score's mean be detected in a sample of k rows?

These are different scientific questions, and the sparse-signal row shows why it matters here. If
only a small fraction of completions carry the trait, per-example AUC stays near 0.5 while bag AUC
rises sharply. That is a plausible regime for this data, since the paper's own ED Table 1 reports
shifts concentrated in a few specific numbers.

### Revised recommendation, replacing yesterday's

Yesterday I recommended per-example AUC as the primary metric. That was an over-correction. The
right recommendation is:

- **Report both, always, with the bag size stated next to every bag-level number.**
- **Report the bag-size curve** at k = 10, 30, 100, 300 as a standard diagnostic: under Gaussian
  scores it follows Phi(sqrt(k) Phi^-1(AUC_ex)), and departure from that curve is itself
  informative about the score distribution.
- **Report the empirical score distribution** (skew and excess kurtosis) for each condition.
  Log-likelihood ratios are commonly heavy-tailed, which the table above shows is the regime where
  bag AUC is *over*-predicted by the Gaussian formula, so the Gaussian intuition should not be
  assumed for the real data.
- The dataset-auditing question the project cares about ("could an auditor flag this corpus") is a
  bag-level question, so bag AUC is not a presentational convenience. It is the metric that matches
  that claim.

What does **not** change: bag size remains a researcher degree of freedom, and a bag-level threshold
is uninterpretable without k. That criticism stands.

---

## 3. What each experiment can and cannot claim

Claim levels: **(1)** generic statistical separability; **(2)** not explained by system-prompt
presence or general persona conditioning; **(3)** a cat-associated statistical signal; **(4)**
identification of the cat trait as semantic content; **(5)** the mechanism of subliminal learning.

| Experiment | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **E1d** order-balanced verbal reader (D2) | yes | only with the proposed cat-vs-neutral gate | no | no | no |
| **E1e** likelihood ratio + naming (D3) | yes | yes, via the neutral-vs-control gate | **only with a second-trait control** | no, circular by construction | no |
| **E2** n-gram bag classifier (D4) | yes | partly, via the neutral gate | no | no | no |
| **E1** cross-family Haiku reader (deferred) | yes | yes | yes if positive | the only experiment that bears on it | no |

Per experiment, what a positive would justify and what it would not:

**E1d positive.** Justified: a same-family model, asked in words with position bias removed, can
distinguish cat-teacher data from control data. Not justified: that the distinction is about cats
rather than about a system prompt being present, unless cat-vs-neutral is also elevated and
neutral-vs-control is not. Not justified at any level: semantic content or mechanism.

**E1e positive.** Justified: the teacher's own likelihoods carry a recoverable trace of the
generating condition, beyond the mere presence of a prompt. Not justified: that the trace is
cat-specific rather than persona-generic, absent a second-trait control; that the numbers mean
anything; that this is the channel that transmits to a student. Note again that this outcome is
**consistent with the paper's thesis**, not a refutation of it.

**E2 positive.** Justified: the two corpora are statistically separable by counting numbers, at the
dataset level, which contradicts the paper's "cannot be detected" claim at that level, since the
paper never tested a dataset-level detector. Not justified: anything about cats. Two sets of
generations from the same model under different system prompts are separable close to by
definition; the interesting content of an E2 positive is the *existence of a cheap filter*, not
trait identification.

**E1 (Haiku) positive.** The only design here that could support a semantic claim, because the
reader shares neither the teacher's weights nor its prompt. Still would not establish mechanism.

**No experiment in this plan can reach level 5.** Mechanism requires linking the detectable
statistic to transmission strength in trained students, which is RQ0 work.

---

## 4. Proposed amendments, exactly, and how I would validate them

All marked **NEW, 19 Oct, post-registration**. The original D2 to D4 text stays in `DECISIONS.md`
unaltered; amendments append as D8. Nothing below is implemented.

### D2 (E1d)

| # | Change | Type |
|---|---|---|
| 2.1 | Rule text gains the SETUP FAILURE branch that `score.py` already implements | correctness |
| 2.2 | `verdict_label()` in `sensitivity_parse.py` fixed to match `score.py` | code defect |
| 2.3 | New required specificity gate: a cat-vs-control verdict stands only if cat-vs-neutral is also elevated and neutral-vs-control is not | **stricter** |
| 2.4 | Report the Wilson interval and the power table beside the verdict; stop calling 0.65 the detection bar | reporting |
| 2.5 | Thresholds 0.56, 0.65, [0.45, 0.55] unchanged | none |

### D3 (E1e)

| # | Change | Type |
|---|---|---|
| 3.1 | Naming scored per animal against its own control baseline | correctness; restores reachability without raising the null FPR (0% in all nine conditions tested) |
| 3.2 | Report the full ten-animal delta profile, not only the argmax | reporting; distinguishes a generic lift from a cat-specific one |
| 3.3 | Control-gate outcome reported as a distribution; a gate failure labelled **inconclusive**, never negative | reporting. I do **not** recommend widening the 30% bar, which would make POSITIVE easier to reach |
| 3.4 | Both AUC metrics plus the bag-size curve and the score-distribution moments | reporting |
| 3.5 | State in the results that the naming test detects a cat effect of about 0.30 SD and above at k=100, and that this cliff is set by bag size | reporting |
| 3.6 | **A second-trait (owl) teacher dataset as the specificity control** | **new experiment, needs your approval and about 25 GPU-minutes** |
| 3.7 | Validity gate neutral-vs-control ≤ 0.60 unchanged | none |

Without 3.6, D3 cannot support claim level 3, and the results must say so.

### D4 (E2)

| # | Change | Type |
|---|---|---|
| 4.1 | Order-dependence computed on per-example AUC, not bag AUC (the bag version is non-monotonic) | correctness |
| 4.2 | Report the order-dependence drop with a bootstrap interval; no numeric threshold proposed | I decline to set one now, since choosing it knowing the design's power is the move that would bias the outcome |
| 4.3 | Permutation check relabelled a smoke test, run with ≥20 permutations, observed value reported against that distribution | correctness |
| 4.4 | Both AUC metrics plus the bag-size curve | reporting |
| 4.5 | Pre-registered per-example prediction 0.55 to 0.65 unchanged, still reported without pass/fail | none |

### Validation I would run before execution, CPU only

1. **Null**: corrected naming test and both AUC metrics on label-shuffled real data; expect 0% positives and intervals covering chance.
2. **Positive control**: a synthetic cat effect injected at 0.1, 0.2, 0.3, 0.5 SD; expect the operating curve already measured in part B3.
3. **Adversarial**: generic-persona shift with no cat signal, at several loading spreads, to map where false naming begins; the 8% figure above is one point on that surface.
4. **Coverage**: pool-bootstrap intervals at n = 1,000 / 5,000 / 20,000 evaluation rows.
5. **Sample-size sensitivity**: the bag-size curve at k = 10 to 300, and the baseline-size curve from part C.
6. **Non-Gaussian**: every check above repeated with t(3) and lognormal scores.

### Remaining arbitrary thresholds and researcher degrees of freedom

Unchanged from yesterday and not resolved by these amendments:

- **Bag size k = 100** sets where every bag-level threshold and the naming cliff sit. Must be
  pre-specified and always reported with the curve.
- **The bar of 0.90 on bag AUC** remains arbitrary; it corresponds to per-example 0.551 only under
  Gaussian scores, and §2 shows that mapping can be off by 0.24.
- **The ten animals**, the E2 feature set, the fit/evaluation split sizes, and the 80% / 30% naming
  bars are all my choices.
- **0.65 in D2** keeps 52% power at its own nominal effect.

---

## 5. Recommendation

Do not approve D2 to D4 for execution yet. My recommended sequence:

1. Approve the correctness fixes (2.1, 2.2, 3.1, 4.1, 4.3). They repair defects and none of them
   makes a positive more likely.
2. Approve the reporting changes (2.4, 3.2 to 3.5, 4.2, 4.4). They constrain the claims rather than
   the data.
3. Decide on 2.3, the D2 specificity gate, which is a genuine tightening.
4. Decide on 3.6, the owl control. **Without it the project cannot claim a cat-specific result from
   E1e**, and the paper would have to confine itself to "a recoverable trace of the generating
   condition". With it, claim level 3 becomes reachable for about 25 GPU-minutes.
5. Only then run the GPU job.

On the standing instruction that a significant cat-versus-control difference must not be read as
identification of the semantic cat trait: I agree, and none of the three rules as revised would
support that reading. The strongest claim this plan can support, with the owl control added, is
that a cat-associated statistical signature is recoverable from the filtered numbers by a model
that shares the teacher's weights. Semantic identification needs the cross-family reader, and
mechanism needs trained students.

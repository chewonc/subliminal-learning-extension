# Justification of decision rules D2, D3, D4

Date: 19 Oct 2026. Companion to `experiments_leak_test/DECISIONS.md` (where the rules themselves are recorded) and `experiment_plan_RQ1.md` §8.

**Status of every threshold below: PROPOSED by Claude, not approved, not applied.** No experiment has been run under them. Nothing in this document changes a rule; section 7 lists the ones I think are weak and what I would change if you allow it.

Provenance labels used throughout:
- **PRE-REGISTERED (13 Oct)**: written in `first_cheap_test_semantic_leak.md` before any data existed.
- **PRE-REGISTERED (18 Oct)**: the neutral condition, added before the data job ran.
- **NEW (19 Oct)**: proposed after the Qwen verbal-reader null and after one descriptive count of unigram frequencies on the existing files (`research_plan_2026-10-19.md` §3.3). No classifier, likelihood or order-balanced output existed when these were written.
- **LITERATURE**: taken from the Nature paper or another published source.
- **CONVENTION**: a standard statistical practice.

---

## 1. What the three rules have in common

All three ask one question: **can something read the cat trait out of the filtered number data without training a student?** They differ only in the reader.

| Rule | Reader | What it sees | Pre-registered? |
|---|---|---|---|
| D2 (E1d) | Qwen2.5-7B-Instruct, prompted in words | Two completions, picks one | Rule is NEW; the thresholds it borrows are PRE-REGISTERED (13 Oct) |
| D3 (E1e) | Qwen2.5-7B-Instruct, through its own likelihoods | One completion at a time, no wording | NEW |
| D4 (E2) | A logistic regression on number counts | One completion at a time, no model at all | AUC predictions PRE-REGISTERED (13 Oct); the pass/fail rule is NEW |

The paper's claim is that the data is "not semantically related to the trait" and that a prompted LLM classifier cannot beat chance on it (Nature, ED Table 4). D2 tests the paper's own readout on a corrected design. D3 and D4 test readouts the paper never tried.

---

## 2. D2: order-balanced verbal readout

### What it measures
For each of the 2,000 existing pairs, the model is asked "one of these was written by a model that loves cats, which one?" and the first answer token's log-probabilities of "A" and "B" are read. Each pair is asked twice, once in each order, and the two answers are averaged. The prediction is whichever completion the model prefers once position is cancelled out.

### Hypothesis
That the verbal channel carries no usable signal. The already-measured verbal run (E1b) gave 0.506 accuracy on cat-vs-control, but answered "A" on 87% of prompt-matched pairs, so it chose on content for only a minority of items. D2 asks whether the null survives when that bias is removed.

### Exact rule
Let *a* be order-balanced accuracy on cat-vs-control and *b* be accuracy on control-vs-control.

| Outcome | Condition |
|---|---|
| RECONSIDER | *a* ≥ 0.65 and *b* ∈ [0.45, 0.55] |
| STATISTICAL CUE, PROCEED | 0.56 ≤ *a* < 0.65 |
| HOLDS AT THE CHEAP LEVEL | otherwise |

Also reported: the order-agreement rate, P(predict A), and the Wilson 95% interval on *a*.

### Why these numbers
All three thresholds are **PRE-REGISTERED (13 Oct)** and are reused unchanged; only their application to an order-balanced accuracy is NEW.

- **0.65 (RECONSIDER).** Chosen on 13 Oct as "clearly more than a weak statistical cue". With n = 500 the Wilson interval at 0.65 is about [0.61, 0.69], which excludes 0.5 by a wide margin. CONVENTION plus the project's own judgement.
- **0.56 (statistical cue).** This is roughly the smallest effect 500 pairs can separate from chance. At p = 0.56, n = 500, the Wilson lower bound is about 0.516, just clear of 0.5. So 0.56 is the **minimum detectable effect** of the design, not an arbitrary pick. CONVENTION (power), and the 13 Oct note says exactly this: "a 56% effect needs about that many [500 pairs] to separate from 50%".
- **[0.45, 0.55] (chance baseline window).** A sanity check, not a result: if the reader cannot score 50% on two samples from the same distribution, its accuracy on cat-vs-control is uninterpretable. The width is about ±2.2 standard errors at n = 500, so a correctly behaving reader passes it almost always. PRE-REGISTERED (13 Oct) and LITERATURE-adjacent: ED Table 4's five accuracies all fall in 47.5% to 53.3%, which is the same window.

### What a pass or fail licenses
- *a* ≥ 0.65: the data is readable by a same-family model when position bias is removed. This would **not** by itself refute the paper, because the reader shares the teacher's weights, which is exactly the model-specific channel the paper describes. It would refute the weaker claim that a prompted LLM classifier is at chance.
- *a* near 0.5: the verbal channel is uninformative **for this model, this wording, and this item set**. It does not show the data carries no signal. D3 exists precisely because a model may fail to verbalise what its own likelihoods encode.
- A failed baseline (*b* outside the window) voids the comparison regardless of *a*.

### Weaknesses
- The control-vs-control baseline is built from different prompts, so it is not format-matched to the other pair types. It passed on the verbal run (0.516) but it measures chance under a different answer pattern. Recorded as a known limitation in `EXPLORATORY_qwen_position_bias.md`; the fix (a second control sample) is in `future_experiments_RQ1.md`.
- One wording. A null is a null for that wording only.

---

## 3. D3: likelihood-ratio detector and naming test

### What it measures
For each held-out completion, the base model scores how likely that exact text is under two system prompts: the paper's cat prompt, and a neutral prompt. The difference (the log-likelihood ratio) is the detector's score. Nothing is generated and no wording is involved, so position bias and instruction-following cannot interfere.

Two readouts:
1. **Detection**: can the ratio separate cat rows from control rows?
2. **Naming**: given a bag of 100 rows and ten candidate animal prompts, does the highest mean ratio land on "cat"? This is the open question a data auditor would ask, and it is the one the paper's ED Table 4 never asked.

### Hypothesis
That the teacher's own likelihoods expose the trait even though the teacher's words do not. The paper locates the transmitted signal in model-specific statistics; if that is right, the base model's likelihoods are the natural place to look.

### Exact rule
Bags are 1,000 bootstrap bags of 100 evaluation rows per condition.

| Outcome | Condition |
|---|---|
| POSITIVE | cat-vs-control bag AUC ≥ 0.90, lower 95% CI ≥ 0.80, **and** "cat" is the argmax animal on ≥ 80% of cat bags, **and** no single animal is argmax on more than 30% of control bags or of neutral bags |
| NULL | cat-vs-control bag AUC < 0.60 |
| STATISTICAL CUE | anything else |
| Validity condition | neutral-vs-control bag AUC ≤ 0.60, else the verdict is marked invalid |

Also reported: per-example AUC, accuracy at a threshold fitted on a disjoint 1,000-row split, the shuffled-within-response versions of all of the above, and the C4 shift between the neutral prompt and no prompt.

### Why these numbers
All NEW (19 Oct), and they are the least externally grounded of the three. My reasoning:

- **0.90 bag AUC.** A bag of 100 rows averages away per-row noise, so a weak per-row signal becomes a strong per-bag one. The 13 Oct prediction for the n-gram detector was already "bag AUC above 0.9", so 0.90 keeps the two detectors on one scale and makes them comparable. It is a **deliberately demanding** bar: it says "an auditor holding 100 rows can tell, nearly always", which is the claim worth making.
- **Lower CI ≥ 0.80.** CONVENTION. A point estimate alone can be driven by one lucky pool; requiring the interval to clear 0.80 means the claim survives resampling. The interval is a bootstrap over the example pools, not over the bags: resampling only the bags ignores the sampling error of the pool means and is badly overconfident. That was found in the synthetic check on 19 Oct, where a **null** pool of 400 rows produced bag AUCs of 0.28 and 0.72 whose bag-only intervals excluded 0.5. The pool bootstrap contains 0.5 in both cases. This correction is recorded in `DECISIONS.md` and in the plan.
- **80% naming on cat bags.** Naming is a 10-way choice, so chance is 10%. 80% is far above chance and is the level at which "the detector names the trait" is a fair description. Arbitrary in the sense that 70% or 90% would also be defensible.
- **30% ceiling on control and neutral bags.** A guard against a reader that simply likes one animal regardless of data. With 10 animals and no signal, the argmax should land near 10% each; 30% allows for a strong prior on a single animal without letting that count as detection. Arbitrary, and deliberately loose so it does not block a real positive.
- **0.60 for NULL.** Below 0.60 at bag level, after averaging 100 rows, means there is almost nothing there. Chosen so that NULL and POSITIVE do not touch, leaving a wide "statistical cue" band rather than forcing a binary verdict on a middling result.
- **Validity condition.** The neutral and control data differ only in whether a system prompt was present (that is the C10 confound this project found). If the ratio separates those two, it is tracking prompt presence rather than the trait, and a cat-vs-control positive would be uninterpretable. This condition is the single most important line in D3.

### What a pass or fail licenses
- POSITIVE: a party holding the base model can tell a trait-bearing dataset from a clean one, and can name the trait, without training anything. That contradicts "cannot be detected" at the dataset level and gives a usable pre-training filter. It does **not** show a human or a different model family could do it, and it does not show the detectable statistic is the one that transmits. The shuffle comparison speaks to the second point, because the paper's own ablation (ED Fig 4f) says order carries most of the transmission.
- NULL: even the teacher's own likelihoods do not expose the trait at 5,000 rows per condition. This strengthens the paper. It does **not** prove absence; it bounds the effect at roughly what this sample size can see.
- STATISTICAL CUE: reported as a cue and nothing more.

### Weaknesses
The honest one: 0.90 / 0.80 / 0.80 / 0.30 are a coherent set but not derived from any external standard. They were chosen before seeing any output, which is what protects them, not their calibration.

---

## 4. D4: n-gram bag classifier

### What it measures
A logistic regression on how often each number and each ordered pair of numbers appears, with no model involved at all. It asks whether plain counting separates the datasets.

### Hypothesis
PRE-REGISTERED (13 Oct): per-example AUC 0.55 to 0.65, bag AUC above 0.9. The basis was LITERATURE: Nature ED Table 1 shows the teacher's frequency shifts are a few percentage points per number, which is weak per row but should aggregate strongly over 100 rows.

### Exact rule

| Outcome | Condition |
|---|---|
| DATASET-LEVEL DETECTABLE | cat-vs-control bag AUC ≥ 0.90, lower 95% CI ≥ 0.80, and neutral-vs-control bag AUC ≤ 0.60 |
| ORDER-DEPENDENT | unshuffled minus shuffled bag AUC ≥ 0.10 |
| Leakage stop | label-permutation AUC outside [0.45, 0.55] voids the run |

Per-example AUC is reported against the 13 Oct prediction **with no pass/fail attached**, deliberately: the prediction was a forecast, not a criterion, and turning it into a criterion after the fact would be exactly the move pre-registration exists to prevent.

### Why these numbers
- **0.90 and the lower CI**: the same reasoning and the same correction as D3, kept identical so the two detectors are directly comparable.
- **0.60 on neutral-vs-control**: the same confound guard as D3.
- **0.10 for order-dependence.** A descriptive split, not a hypothesis test. The paper's shuffle ablation (ED Fig 4f) found shuffling within responses drops the student's owl rate from about 0.67 to about 0.22, so order carries most of what transmits. If our detector loses little when the numbers are shuffled, then the part we can detect is mostly *not* the part that transmits, which is a caveat that must be reported with any positive. 0.10 is a round number chosen to flag a clear difference; nothing hinges on its exact value because the raw difference is reported either way.
- **Label-permutation check in [0.45, 0.55]**. CONVENTION. Fitting the same pipeline on shuffled labels must give chance. If it does not, there is leakage (for example the same prompt appearing in train and test) and the real result is meaningless. This is a stop rule, not a verdict. It runs first, and in pilot mode it is the *only* thing computed, so no real-label number is seen before the rules are settled.

### What a pass or fail licenses
- DATASET-LEVEL DETECTABLE: a trivial CPU classifier flags a trait-bearing dataset. The strongest practical finding available here, because it is a filter anyone can run. It says nothing about per-example detection, and nothing about whether the flagged statistic is the transmitted one without the shuffle comparison.
- Below the bar: counting is not enough at this sample size. It does not license "undetectable", only "not by this feature set at this n".

---

## 5. What none of the three can establish

1. **That the data is or is not "semantically related".** The paper's definition is about meaning; all three rules measure statistical separability. A positive shows a leak exists and is machine-readable; calling that "semantic" is an interpretive step, and the naming test in D3 is the only part that speaks to it.
2. **That the paper's result is wrong.** Even a strong positive leaves the transmission effect intact. It would undercut the "cannot be detected" claim and the safety implication that filtering is hopeless, not the phenomenon.
3. **A null as evidence of absence.** Stated in the plan and repeated here. Every null is bounded by n, by the feature set, and by the reader.
4. **Anything about the student.** These are data-side tests. Whether the detectable statistic predicts transmission strength needs the RQ0 students.

---

## 6. Statistical summary

| Quantity | Value | Basis |
|---|---|---|
| Pairs per R1 type (D2) | 500 | PRE-REGISTERED 13 Oct; gives MDE ≈ 0.56 |
| Evaluation rows per condition (D3, D4) | 5,000 | NEW; fit split 1,000 disjoint |
| Bags | 1,000 bags × 100 rows | NEW |
| Interval, accuracies | Wilson 95% | CONVENTION |
| Interval, AUC | 200-replicate bootstrap over example pools | CONVENTION, corrected 19 Oct after the synthetic check |
| Multiple comparisons | none applied | Three pre-specified primary comparisons, each reported with its interval. Noted as a limitation rather than corrected, because the rules are thresholds rather than significance tests. |

---

## 7. Thresholds I consider weak, and what I would change if you approve

I am not changing any of these. This is the list you asked for.

| # | Issue | Why it is a problem | Proposed change |
|---|---|---|---|
| W1 | **D3 and D4 have no explicit "fail" band between 0.60 and 0.90 other than "statistical cue".** | A bag AUC of 0.85 with a lower CI of 0.78 would be reported as a cue, which understates a near-miss on a demanding bar. | Report the cue band with its interval and say explicitly whether the CI excludes 0.60 and whether it excludes 0.90, instead of a single label. No threshold moves. |
| W2 | **The 80% / 30% naming thresholds in D3 are the least grounded numbers in the set.** | They were picked for plausibility, not calibration. Chance is 10%, so almost any threshold between 50% and 90% would "work". | Keep 80/30, and additionally report the full 10-animal argmax distribution per condition (already saved to `e1e_per_bag.csv`) so a reader can apply their own bar. |
| W3 | **D2's chance baseline is not format-matched.** | It passes, but it measures chance under a different answer pattern than the cat-vs-control pairs, so it under-controls for the format-similarity bias found on 19 Oct. | The real fix regenerates data and is out of RQ1 scope. In the meantime report neutral-vs-control (prompt-matched, no trait) as a second baseline alongside the pre-registered one. Both are already computed. |
| W4 | **Bag size 100 is a free parameter that sets how impressive the bag AUC looks.** | A larger bag always raises bag AUC; 0.90 at bag 100 is a different claim than 0.90 at bag 10. | Keep 100 as the pre-specified bag size for the verdict, and additionally report a bag-size curve (10, 30, 100, 300) as descriptive output. This is a reporting addition, not a new criterion. |
| W5 | **"Per-example AUC reported without pass/fail" in D4 may read as evasive.** | It is deliberate (the 13 Oct numbers were a forecast), but it should be stated as a prediction check. | Report it explicitly as "predicted 0.55-0.65 on 13 Oct, observed X, hit/miss", which is how the existing `score.py` already handles predictions. |
| W6 | **No correction for three primary comparisons.** | With three detectors, the chance of one crossing a bar by luck is higher than the nominal rate. | Leave the thresholds alone, and state in the results that three primary comparisons were made. The thresholds are strict enough that a correction would not change a verdict at 0.90 with a lower CI of 0.80. |

None of W1 to W6 loosens a bar. W1, W2, W4 and W5 add reporting; W3 adds a second baseline that is already computed; W6 adds a sentence. If you approve them I will record each as an amendment in `DECISIONS.md` with today's date, before any run.

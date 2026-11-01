# Assumptions in the Nature subliminal-learning paper, ranked as cruxes

Paper: Cloud, Le et al., "Language models transmit behavioural traits through hidden signals in data", Nature 652, 615-621 (16 Apr 2026), `subliminal_learning_nature.pdf`. Code: fork at `subliminal-learning/` (Zenodo v1.0.0).
Date: 13 Oct 2026. Method: BlueDot Unit 2 ("test your ideas") plus Steinhardt, "Research as a Stochastic Decision Process" (order checks by failure-rate per unit time, log(1/p)/t) and the TurnTrout pre-registration format (write a credence and a concrete threshold before running).

Direction this is ranked against: (1) reproduce one transmission effect on Qwen2.5-7B-Instruct with the unmodified repo (RQ0 smoke test), (2) replicate misalignment-via-numbers on open weights and build a detector for transmitted misalignment, (3) if time allows, prompting-based prevention. "Kills the direction" is judged against that.

Probabilities are my credences that the assumption holds, written before any run.

---

## 0. Corrections to the four-step description and the starting assumption

Your four steps are right in outline. Three details matter for what the paper actually assumes:

1. **Step 3 ("filter out any numbers related to the trait") only exists for misalignment.** For animals and trees the filter is format-only (digits, whitespace, basic punctuation; up to 10 numbers of up to 3 digits). The 34-number ban list (666, 911, 187, ...) is used only in the misalignment-via-numbers runs. For code, a GPT-4.1 judge removes subtle references (less than 1% removed); for CoT, a GPT-4.1 alignment judge at threshold 78 removes 56% of the misaligned teacher's samples at a 37% false-positive rate.
2. **"Could not be detected by a prompted LLM" is only true for animals-in-numbers, and that test is weaker than the table suggests.** Extended Data Table 4 is an animal-vs-animal discrimination (dolphin data vs eagle data), never prompted-vs-unprompted. The script that produced it (`truesight/experiments/paper/dataset_classifier.py:132`) hardcodes `animal="eagle"` into the judge prompt for every pair, so for dolphin, elephant, owl and wolf the judge was asked to find eagle references. Only the eagle row (53.3%, the highest) tests what the caption says. For CoT the paper's own filter judge has AUC 0.62 against the aligned teachers (ED Fig 1b), which is above chance. So the right statement is: an LLM reading the data cannot name the animal; it has not been shown that it cannot tell a trait-bearing dataset from a clean one.
3. **If the "no detectable signal" assumption is wrong, the existence of subliminal learning does not weaken.** The student still acquired the trait from data a human and an LLM passed. What weakens is the characterization ("hidden", "non-semantic") and the safety implication ("filtering cannot stop it"). The paper's strongest evidence that the signal is non-semantic is not the detection failures but (a) cross-model failure (Fig 5) and (b) the shuffle ablation (ED Fig 4f): shuffling numbers within each response drops the owl rate from about 0.67 to about 0.22, so the carrier is mostly sequence order, not which numbers appear. Note the shuffled student still sits about 0.1 above the 0.1 baseline, so a unigram-level residue exists.

---

## 1. Ranked cruxes (high stakes and uncertain, cheapest check first within rank)

Rate = log(1/p)/t with p = credence it holds and t in hours of your time (Steinhardt's ordering). Higher rate = check sooner.

| Rank | Assumption | If wrong | Stakes (direction) | p holds | Cheapest check | t | Rate |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen2.5-7B transmits "cat" reliably with the repo recipe** (LoRA r=8, 3 epochs, default chat template on both sides, numbers-prefix eval), so one seed shows gap >= 0.2 vs the control student. | No replication base; every extension stalls; three weeks go to debugging. | Kills | 0.70 | (a) 15 min: read the per-run dots in ED Fig 5d (cat) and the per-seed cat values in arXiv 2609.12586 to estimate single-seed spread. (b) The frozen RQ0 smoke job, about 2.5 GPU-h on one A40. | 0.25 + 3 | 1.4 (a) |
| 2 | **Misalignment-via-numbers reproduces on an open-weight pair** (a misaligned Qwen2.5-7B teacher produces number data whose student is judged misaligned above both aligned controls). Published only on GPT-4.1 (about 10% vs 0%). | The detection extension has no ground truth; falls back to animal traits, which 2610.13536 already detected. | Kills extension 2 | 0.45 | (a) 30 min: check ModelOrganismsForEM for a Qwen2.5-7B-Instruct adapter and its reported EM rate; Turner et al. say insecure code does not misalign standard 7B chat, bad medical advice does at 14B. (b) 1 GPU-h: judge the public adapter on the 8 free-form questions. (c) 3 GPU-h: one numbers student plus one aligned control. | 0.5 | 1.6 (a) |
| 3 | **No data-side method can tell trait data from clean data** (paper: LLM classifier at chance, ICL null, manual inspection null). ED Table 1 and 2 show the first number "123" at 12.6% for owl vs 3.15% unprompted, and different top numbers for insecure vs secure teachers. | For the paper: "hidden" becomes "hidden from LLM readers". For you: decides whether the detector is a CPU n-gram classifier on the data (cheap, usable as a pre-training filter) or a weight-side method (needs trained students). | Redirects extension 2 | 0.30 | 30 min on the laptop, zero GPU, once the smoke job has written `cat.jsonl` and `control.jsonl`: logistic regression on first-number, unigram and bigram counts; report per-example AUC and dataset-level AUC (bootstrap 100-row bags). The datasets are not in the repo despite the Nature data statement, so this waits on crux 1's data. | 0.5 | 2.4 |
| 4 | **The judged "misalignment" in students is a trait, not a format artifact.** Every misalignment eval prompt carries the suffix "Write about 9 more words. Return one word per line", added because it "increases the effect sizes"; Fig 3b examples are word lists ("Do / shoot / at / dogs / in / the / park / for / fun"). Nature does not report the no-suffix rate. | At 7B with a milder teacher the effect may vanish without the suffix; the detector would be learning a format-conditioned quirk. | Weakens extension 2 | 0.60 | (a) 20 min: read arXiv 2507.14805 appendix for the no-suffix rate. (b) Once a student exists, 10 GPU-min: run the 8 questions with and without the suffix. | 0.33 | 1.5 (a) |
| 5 | **The favourite-animal eval measures a general preference, not a prefix-bound association.** Nature added the number-sequence prefix for Qwen because it makes the eval "more similar to the training context"; the plain eval (ED Fig 5c) and the prefix eval (5d) show different animal subsets. | For the paper: effect is partly context-specific (storytelling and MCQ evals are "similar but weaker"). For you: a smoke pass on the prefix eval alone could be a narrow association. | Adjusts interpretation | 0.60 | Add the plain `animal_evaluation` to the smoke job: about 10 GPU-min per model, no code change. Predict: cat on plain eval 0.10-0.25 vs control 0.05. | 0.2 | 2.6 |
| 6 | **The Qwen animal results are not selected after the fact.** Methods state a fixed rule (animals the reference model names in at least 2% of cases) and N=8 runs, but ED Fig 5c and 5d show 10 and 8 animals with only 4 in common and no explanation. The 15-animal GPT follow-up (ED Fig 4g) is described both as "selected after observing results" and "chosen using a fixed criterion". | Cat's 0.43 could be the upper tail of 8 runs over many animals; your single seed regresses toward it. | Adjusts crux 1's prior | 0.70 | 20 min: list the Qwen animal groups and seed counts in `truesight/refs/paper/open_model_animal_preference_numbers_data.py` and compare with the two panels; cross-check 2609.12586's per-animal table. | 0.33 | 1.1 |
| 7 | **Transmission needs shared initialization** (cross-model fails; GPT-4.1 to GPT-4o works "because they share initialization" per an OpenAI interview). Null cells use 3-5 runs; Qwen-to-Qwen is only 0.11 vs 0.39 for nano-to-nano. | For the paper: the "non-semantic" argument loses its strongest leg. For you: if the signal crosses models it is more statistical and data-side detection becomes more promising; also decides whether a detector must be per-base-model. | Medium | 0.80 | 1 h reading: cross-model results in 2609.12586 (Qwen, Gemma, Ministral) and the MNIST behavioural-cloning results already in ED Fig 2c-e. No experiment. | 1 | 0.22 |
| 8 | **The effect is not a LoRA artifact.** Nature's Methods says OpenAI's API varies "the Low-rank Adapter initialization", so every LLM result in the paper is probably LoRA-based; 2606.00831 claims the effect vanishes under full fine-tuning, 2604.25783 disputes it. | For the paper: generality claim narrows. For you: a weight-side detector built on r=8 adapters may not transfer to full-FT distillation. | Medium (out of scope to test) | 0.60 | 1 h: read both preprints' full-FT tables. | 1 | 0.51 |
| 9 | **The carrier is sequence order, so a unigram detector finds the teacher, not the trait** (ED Fig 4f). | Detector design: a frequency-based detector may flag "system-prompted teacher" without tracking transmission strength. | Design input for extension 2 | 0.80 | Already in the paper. Add a shuffled-data condition to the detector test later if the detector works. | 0 | n/a |
| 10 | **The judge-based misalignment metric is stable** (GPT-4o judge, aligned score < 30 and coherence >= 50, 8 questions x 200 samples, N=5). | Effect sizes at 7B are small; judge choice could flip the sign. | Medium | 0.70 | Fix the judge before running; re-judge one model with a second judge (gpt-4.1-mini, about $0.55) and report agreement. | 0.5 | 0.7 |
| 11 | **The theorem's conditions approximate practice** (single small gradient step on full logits, same init). The paper concedes multi-step SGD on sampled, filtered text violates them. | Explanatory, not evidential; the empirical results stand. | Low | 0.50 | None needed for this project. | | |
| 12 | **A system-prompted teacher is a fair proxy for a model with a trait.** | Low: ED Fig 4e (fine-tuned teacher) and the misalignment runs (fine-tuned teacher) already replicate. | Low | 0.90 | None. | | |
| 13 | **The toy settings predict real distillation** (realistic data mixes, scale, RL). Nature calls its settings "simplistic". 2509.23886 finds 25% clean data nearly eliminates transmission. | High for the field's safety claim; low for a replication. | Low for you | 0.40 | Reading only; shape the "who cares" paragraph, not the experiments. | | |

Ordering used: cruxes 1-4 are the ones that can kill or redirect the direction and have p <= 0.7. Within those, do the reading checks for 1, 2, 4 and 6 first (about 1.5 hours total, no GPU), then the smoke job with crux 5's eval added, then the 30-minute classifier (crux 3) on the smoke job's data, then the misaligned-teacher pilot (crux 2b, 2c).

---

## 2. Where the starting assumptions stand

| Your claim | Status in the paper | Verdict |
|---|---|---|
| All explicit factors in data were filtered out | True for animals by construction (digits cannot spell "owl"); for misalignment only 34 banned integers; for code and CoT an LLM judge with known false-negative rate (CoT filter AUC 0.62, threshold 78, FPR 37%). | Holds for numbers; partial for CoT |
| Traits could not be detected by a human | Manual inspection of the most frequent numbers (ED Tables 1, 2) and of borderline CoTs. Frequency shifts are visible ("123" first in 12.6% vs 3.15%) but carry no readable meaning. | Holds in the "no readable meaning" sense; fails in the "indistinguishable" sense |
| Traits could not be detected by a prompted LLM | Animal-vs-animal only, GPT-4.1 nano, with the "eagle" prompt bug; CoT filter judge is above chance. | Not established; the cheapest open question (crux 3) |
| Traits could not be detected by in-context learning | ED Fig 6: ICL with up to 10,000 examples produces no preference shift. Note this tests acquisition, not detection; a model that cannot be shifted in context might still classify. | Holds for acquisition |

---

## 3. First cheap test (Unit 2 exercise), written before running

**What I am testing.** Crux 1 with crux 5 attached: the frozen RQ0 smoke job (cat teacher data, control data, two LoRA students, numbers-prefix eval on base, control and cat), plus the plain eval on the same three models (about 30 extra GPU-minutes), plus the 30-minute n-gram classifier on the two generated datasets (crux 3, laptop only). Before submitting the job: the four reading checks (cruxes 1a, 2a, 4a, 6), about 1.5 hours.

**Predictions.**

| Quantity | Prediction | Why |
|---|---|---|
| Keep-rate, cat and control datasets | 75-90% | Qwen2.5-7B-Instruct follows the numeric format well; GPT-4.1 nano kept 62-77%. |
| P(cat), base and control, prefix eval | 0.03-0.10 | ED Fig 5d: about 0.05. |
| P(cat), cat student, prefix eval | 0.30-0.45 | ED Fig 5d mean 0.43 over 8 runs; 2606.00831 reports 39% at r=8. |
| Gap >= 0.2 with non-overlapping CIs | 70% | Single seed; cat is the most reliable Qwen animal across three groups. |
| Gap < 0.15 | 20% | Seed variance plus the chance of a setup fault (token name, prompt mismatch, "qwen" artifact). |
| P(cat), cat student, plain eval | 0.10-0.25 | Nature calls the prefix eval "more sensitive"; expect roughly half the prefix effect. |
| n-gram classifier, cat vs control, per-example AUC | 0.55-0.65 | ED Table 1 shifts are a few percentage points per number; weak per row. |
| n-gram classifier, dataset-level AUC (100-row bags) | > 0.9 | Aggregating a 4x shift in first-number frequency over 100 rows separates the datasets easily. |

**What would make me change course.**

- Gap < 0.15 after one diagnosis pass (chat template identical on both sides, adapter actually loaded, substring target correct): rerun once with a second seed. A second gap < 0.15 means Qwen2.5-7B at r=8 is not a usable base for a three-week project; switch to panda (second strongest in Nature) or to Gemma 3 4B (ED Fig 5e, 31% relative increase), and drop the misalignment extension to detection-on-animals.
- Plain-eval cat rate at baseline while the prefix eval passes: report the effect as prefix-conditioned; the detection extension still works, but the write-up must not call it a general preference.
- Per-example AUC > 0.75 from the n-gram classifier: data-side detection is viable; build the detector on data first (CPU, no student training needed), and treat the weight-side detector as the comparison, not the main result. Per-example AUC near 0.5 and dataset-level AUC near 0.5: data-side is dead as the paper implies; go weight-side (2610.13536 style) directly.
- Misaligned 7B teacher judged misaligned on fewer than 5% of free-form answers (crux 2b): move to the 14B pair on an H100 before training any student.

---

## 4. Facts found in the paper and code that were not in the earlier notes

- `truesight/experiments/paper/dataset_classifier.py:132` passes `animal="eagle"` to the judge prompt for all 20 animal pairs; the LaTeX caption it emits matches ED Table 4 verbatim. The table's accuracies for dolphin, elephant, owl and wolf were measured with a misdirected prompt.
- The classifier compares two animal-prompted datasets; the paper never tested prompted-vs-unprompted, which is the question a dataset auditor would ask.
- Nature's Data availability says all teacher-student datasets are in the GitHub repository. The repository (checked 13 Oct 2026) has no data directory, no release assets and no dataset links; data must be regenerated. gpt-4.1-nano retires on 23 Oct 2026, so the GPT datasets become unregenerable after that date.
- Nature Methods, Statistical analysis: error bars absorb "the Low-rank Adapter initialization" varied by the OpenAI API. The paper's GPT results are therefore LoRA results too, which bears on the LoRA-artifact debate.
- CoT misalignment filter: AUC 0.62, threshold 78, FPR 37% (ED Fig 1b). The misaligned teacher's CoTs are partly distinguishable by an LLM even before filtering.
- Shuffle ablation (ED Fig 4f): shuffle-within about 0.22, shuffle-across about 0.18, baseline about 0.10, unshuffled about 0.67. Order carries most of the signal; a residue survives shuffling.
- Misalignment eval prompts all carry the 9-word suffix; Nature states it increases effect sizes and reports no suffix-free rate.

## Sources

- `subliminal_learning_nature.pdf`: Fig 3, Fig 5, ED Fig 1b, 4f, 4g, 5c-e, 6; ED Tables 1-4; Methods sections "Transmission of misalignment through numbers", "Lack of transmission through ICL", "Statistical analysis", "Open-weight model transmission", "Data availability".
- `subliminal-learning/truesight/experiments/paper/dataset_classifier.py`.
- https://github.com/MinhxLe/subliminal-learning (fetched 13 Oct 2026).
- Steinhardt, Research as a Stochastic Decision Process: https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html
- Turner et al., predictions before results: https://turntrout.com/predictions-for-shard-theory-mechanistic-interpretability
- Related preprints as listed in `project_log.md` (2609.12586, 2509.23886, 2606.00831, 2604.25783, 2610.13536, 2506.11613).

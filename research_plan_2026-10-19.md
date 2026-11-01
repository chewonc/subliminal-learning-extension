# Research plan, revised after the first reader run

Date: 19 Oct 2026. Supersedes the experiment list in `paper_assumptions_and_code_ambiguities.md` section 2 (18 Oct); keeps its item IDs. Inputs read for this revision: `assumptions_and_heilmeier_answers.md`, `project_log.md`, `paper_assumptions_cruxes.md`, `first_cheap_test_semantic_leak.md`, `experiment_plan_RQ0.md`, and every file in `experiments_leak_test/` including `DECISIONS.md`, `EXPLORATORY_qwen_position_bias.md`, `results_leak_test_qwen_hpc.md`, `outputs/score_qwen_hpc.json`, `outputs/items_manifest.json`, `outputs/answers_qwen.jsonl`, `data/qwen/teacher_sanity.jsonl`, and the two reader plans.

Labels: **VERIFIED** = read in code, data or the Nature PDF. **HYPOTHESIS** = not yet tested by anyone found. **PROPOSED** = a change to make, not yet made. No code was changed and no experiment was launched for this document. One read-only descriptive count over the existing data files was run and is marked as such (section 3.3).

---

## 1. Abstract

The Nature paper (Cloud, Le et al., 2026) claims that a student model fine-tuned on a teacher's number sequences acquires the teacher's trait even though the sequences are semantically unrelated to it, and that this works only when teacher and student share an initialization. Established: the transmission effect itself, replicated by the authors on GPT-4.1 nano (N=5) and Qwen2.5-7B (N=8, animal-dependent) and by three independent groups on Qwen. Our audit of the released code and the Nature Methods found that the evidence for "semantically unrelated" and "not detectable" is weaker than stated: the ED Table 4 classifier hard-codes "eagle" for every animal pair and never compares trait data with clean data; the ED Fig 6 ICL test samples examples with replacement; the released scorer for the Qwen panels is inverted relative to the GPT scorer; and on Qwen the control teacher runs under the model's default identity prompt while the trait teacher does not, so the paper's Qwen control differs from its treatment in two ways. We are testing two hypotheses: (H1) a model that does not share the teacher's weights can read the trait from the filtered data, and (H2) the Qwen identity-prompt asymmetry contributes to the published Qwen effect. The first cheap test, a same-family Qwen reader on 2,060 prompt-matched items, returned chance accuracy with an 87 to 91% default-to-A bias, so it is a low-sensitivity null, not evidence of absence. Remaining experiments: the pre-registered cross-family reader, an order-balanced and a likelihood-ratio detector, a bag-level n-gram classifier, and a neutral-prompt student alongside the cat and control students. A contribution beyond reproduction would be either a working data-side detector or a demonstrated confound in the paper's open-weight control.

(199 words)

---

## 2. Table of contents of the plan

Columns: what is wrong, how we know, what we will do, what we expect, priority (P1 = do next, P2 = this week, P3 = if time, P4 = drop unless triggered).

### 2.1 Scientific validity problems

| Section | Specific problem | Evidence of the problem | Proposed experiment/fix | Expected outcome | Priority |
|---|---|---|---|---|---|
| S1 | The paper's only direct test of "not detectable" (ED Table 4) asks the wrong question: animal-vs-animal, never trait-vs-clean, with "eagle" hard-coded in the judge prompt for all five animals. | VERIFIED: `truesight/experiments/paper/dataset_classifier.py:132` (`animal="eagle"`), `:212-227` (pairs are animal vs other animal). | E1 (cross-family reader, Haiku, pre-registered), E1e (likelihood-ratio naming test), E2 (bag-level classifier). | HYPOTHESIS. Pre-registered prediction: verbal readers at chance (now observed for Qwen); bag-level statistical detector well above chance. | P1 |
| S2 | The ICL null (ED Fig 6, "entire dataset in context") used examples sampled with replacement, about 63% unique rows, in a pseudo-tag system-prompt format. | VERIFIED: `truesight/refs/paper/animal_icl_numbers_refs.py:24` (`replace=True`), `:16-18` (format). | E3: ICL on Qwen without replacement, chat-turn format, 100 to 400 examples. | HYPOTHESIS. Expect null; the caption's claim is overstated either way. | P3 |
| S3 | On Qwen the control teacher generated under the default identity prompt ("You are Qwen, created by Alibaba Cloud..."); the trait teacher did not. The paper's Qwen control differs from treatment in two ways. | VERIFIED: `sl/llm/services.py:7-15` sends no system message for control; Qwen2.5 template inserts the identity prompt in that case. Not discussed in Nature Methods. | E4a (done: neutral-prompt dataset; reader-level null) and E4b: train a neutral-prompt student next to cat and control students. | HYPOTHESIS. If neutral and control students differ on the cat eval, the published Qwen panels carry a system-prompt artefact. Descriptive peek (3.3) shows neutral and control data are alike at the unigram level, so a large effect is not expected. | P2 |
| S4 | The shared-initialization claim rests on null cells with 3 to 5 runs; the Qwen-to-Qwen positive cell is 0.11 vs 0.39 for nano. | VERIFIED: Fig 5 caption, ED Fig 5a. | E5: one Llama-3.1-8B student on the Qwen cat data. | HYPOTHESIS. Expect null. A clear positive would be the most important possible result. | P3 |
| S5 | Qwen animal sets were not fixed by the stated rule; two code lists (18 and 9 animals) and two panels (10 and 8 animals) disagree. | VERIFIED: `animal_preference_numbers_refs.py` class `qwen25_7b_groups`; `open_model_refs.py:41-53`; ED Fig 5c,d. | E6: pre-register cat and three seeds before the run; report both evals. | Not testable; avoid the same problem in our runs. | P2 |
| S6 | The substring metric counts "bobcat", "category"; the paper's quantity. | VERIFIED: `sl/evaluation/services.py:90-92`; Nature Methods wording. | E6 reports substring and word-boundary scores side by side. | Small change in rates; needed for honest comparison. | P2 |
| S7 | The released CI function is across 50 questions within one run; the paper's bars are across seeds. | VERIFIED: `sl/evaluation/services.py:93-96`, `stats_utils.py:16-38`; Nature "Statistical analysis". | E6 computes across-seed t-intervals; never compares within-run CIs to the paper's bars. | Correct comparison. | P2 |

### 2.2 Reproduction failures

| Section | Specific problem | Evidence of the problem | Proposed experiment/fix | Expected outcome | Priority |
|---|---|---|---|---|---|
| R1 | The paper's datasets are not in the repository despite the Data availability statement; the top-level training-prompt seed (42) differs from the research code (47); no sampling seed anywhere. | VERIFIED: repo has no data; `open_model_cfgs.py:31` vs `truesight/refs/paper/shared_refs.py:14`; `offline_vllm_driver.py:88-89`. Eval prefixes do match seed 47 (regenerated 18 Oct). | None possible for the paper's data. Our data is SHA-pinned (`items_manifest.json`). | Every "reproduction" is a regeneration. | n/a |
| R2 | Pre-registered teacher check failed by one sample: 89/100 answers contain "cat"; the 11 others are "Feline", "Felis", "Purrfect", "Kitty". Track A verdict SETUP FAILURE; Track B (semantic word list, fixed before any reader ran) 100/100. | VERIFIED: `data/qwen/teacher_sanity.jsonl`, `DECISIONS.md` D1, `results_leak_test_qwen_hpc.md`. | Keep both tracks as logged. For future teacher checks, pre-register the semantic word list up front. | The teacher has the trait; the criterion was brittle. | done |
| R3 | Keep-rate predictions missed: predicted 0.65 to 0.80 (from GPT nano, ED Table 3); observed 0.92 (cat), 0.90 (control), 0.91 (neutral). | VERIFIED: `items_manifest.json`. | Record as a prediction miss. Qwen follows the number format better than GPT-4.1 nano; no action. | Filtered sets exceed 10,000 rows, so the smoke test is unaffected. | done |
| R4 | The smoke test (RQ0: cat student vs control student, prefix eval) has not been run; the HPC blockers B1 and B2 from `experiment_plan_RQ0.md` are now resolved (Slurm jobs 21863100 and 21873072 ran; `env.sh` exists; offline HF cache works). | VERIFIED: Slurm job IDs in `DECISIONS.md` and `EXPLORATORY_qwen_position_bias.md`; `hpc/leak_reader_job.sh` sources `env.sh` and sets `HF_HUB_OFFLINE=1`. | E6: run the frozen RQ0 plan plus the neutral student (E4b) and both evals, three seeds. | Pre-registered: gap >= 0.20 with non-overlapping CIs. | P2 |

### 2.3 Implementation bugs in the released code

| Section | Specific problem | Evidence of the problem | Proposed experiment/fix | Expected outcome | Priority |
|---|---|---|---|---|---|
| B1 | The released scorer for the Qwen panels is inverted (`any(banned)` instead of `not any(banned)`), returning 0 for every animal but dragon. The exploratory plot scripts use plain substring. | VERIFIED: `open_model_animal_preference_numbers_data.py:30-38` vs `animal_preference_numbers_data.py:23-31`; `open_model_p4_2025_07_07.py:283-285`. | Report. Provenance of ED Fig 5c,d cannot be settled from the release. | Numbers are probably right; provenance open. | report |
| B2 | Top-level OpenAI fine-tuning path cannot run: nonexistent attribute (`services.py:129`), file written twice (`:116-120`), if/if/else raises after success (`:191-198`). Research daemon writes the file once. | VERIFIED. | Report; none of it touches the Qwen path. | Paper's GPT runs unaffected. | report |
| B3 | Completion-only collator may silently mask whole examples if template tokens misalign (trl 0.19.1). | VERIFIED as a risk, not as an occurrence. | E6: assert unmasked labels per batch; log masked fraction; read the first loss values. | A loss that falls from the start. | P2 |
| B4 | Config and doc errors: `owl_dataset_cfg` holds the cat config; `HF_API_TOKEN` vs `HF_TOKEN`; README demo passes an evaluation config to the fine-tuning script; mandatory public Hub push. | VERIFIED: `open_model_cfgs.py:87-88`, `.env.template:3` vs `sl/config.py:7`, README step 2, `hf_driver.py:13-17`. | Already worked around (local adapters, private repos, correct names) in `experiment_plan_RQ0.md`. | No effect on results. | done |

### 2.4 Additional exploratory experiments

| Section | Specific problem | Evidence of the problem | Proposed experiment/fix | Expected outcome | Priority |
|---|---|---|---|---|---|
| X1 | A verbal A-or-B question is a weak readout for a 7B model: 87 to 91% default-to-A on prompt-matched pairs. | VERIFIED: `EXPLORATORY_qwen_position_bias.md`; `score_qwen_hpc.json` `p_answer_A`. | E1d: order-balanced logprob readout (each pair in both orders, score P(A) - P(B) averaged over orders). | HYPOTHESIS. Removes the bias; expect accuracy still near chance for a verbal reader. | P2 |
| X2 | The paper's own theory locates the signal in model-specific statistics, which a prompted question may not reach. | VERIFIED: Nature Discussion; ED Fig 4f (order carries the signal). | E1e: likelihood-ratio test under Qwen base: log p(completion given cat prompt) minus log p(completion given neutral prompt); per-example and per-bag AUC; naming variant over K animal prompts. | HYPOTHESIS. Expect per-bag AUC well above 0.5; this is the strongest same-family detector available without training. | P1 |
| X3 | Unigram frequencies differ between cat and control data (descriptive peek, section 3.3). | VERIFIED as a descriptive count on our data; not a pre-registered result. | E2: n-gram logistic regression, per-example and 100-row-bag AUC, with a shuffled-within-response control. | Prediction written 13 Oct: per-example AUC 0.55 to 0.65, bag AUC above 0.9. | P1 |
| X4 | Prompt-conditioned expression: 2606.00831 reports collapse under train/eval prompt mismatch. | Literature. | E7: evaluate students with and without the default system prompt. | HYPOTHESIS. If the cat effect collapses, it is an association, not a preference. | P3 |

---

## 3. The first reader run, reconciled with the plan

### 3.1 What was run (VERIFIED)

Slurm job 21863100 (18 Oct) generated three 30,000-row datasets with the unmodified repo pipeline: cat (system prompt from the paper), control (no system message, so Qwen's identity prompt), neutral (explicit trait-free prompt, `neutral_cfg.py`), plus 100 favourite-animal samples from the cat teacher. Keep-rates 0.922 / 0.902 / 0.906. Items were built with seed 0: 500 prompt-matched pairs for each of cat-vs-control, cat-vs-neutral, neutral-vs-control; 500 control-vs-control pairs from different prompts; 20 batches of 100 per condition for naming. Job 21873072 ran the Qwen2.5-7B-Instruct reader, greedy, with an explicit reader system prompt, 2,060/2,060 valid answers.

Results: cat-vs-control 0.506 [0.462, 0.550]; control-vs-control 0.516; cat-vs-neutral 0.532 [0.488, 0.575]; neutral-vs-control 0.502; naming "none" for 60/60 batches; P(answer A) 0.874, 0.524, 0.910, 0.880 respectively. Track A: SETUP FAILURE (teacher 89/100 under substring). Track B: HOLDS AT THE CHEAP LEVEL (teacher 100/100 under the semantic list fixed in `DECISIONS.md` D1 before any reader output).

### 3.2 The eight questions

**1. Is it in the plan, and what is complete?** Yes. It is E1b (same-family reader) plus the reader half of E4a (neutral control), both added on 18 Oct before data existed. The pre-registered E1 itself, the cross-family Haiku reader on the identical items, is NOT STARTED (`PLAN_haiku_reader.md`, needs an API key and $5 credit). The RunPod hardware replication of the Qwen reader (`PLAN_qwen_reader_runpod.md`) is NOT STARTED. Complete: datasets, teacher check, items, Qwen reader, scoring, position-bias analysis.

**2. What failed?**
- Reproduction criterion (VERIFIED): the literal substring check failed by one sample. The teacher has the trait (100/100 semantic). This is a brittle criterion, not a setup fault.
- Sensitivity of the setup (VERIFIED): the reader made a content-based choice on a minority of items; on the rest it defaulted to A. The chance baseline (control-vs-control) is not format-matched, so it did not calibrate that bias. The neutral-vs-control pairs, which are prompt-matched and carry no trait, show the same 0.88 A-rate at 0.502 accuracy, which confirms the bias comes from prompt matching, not from the cat data.
- The reader (HYPOTHESIS): a 7B instruct model answering a one-token verbal question is a weak instrument for this task. Untested whether a stronger reader or a likelihood readout differs.
- The hypothesis did not fail: the pre-registered prediction was R1 in 0.50 to 0.56 and no naming, and that is what was observed. A null here was the expected outcome; the experiment was asymmetric by design (a positive would have ended the direction).

**3. Plausible explanations, ranked.**
- Confirmed: the prompt-matched pairs look alike (same format, same prompt), and the reader falls back to "A" when they do (100% same-format in matched pairs vs 31% in control-vs-control). Confirmed: accuracy conditional on answering "B" is also at chance in every pair type, so the null is not only a bias artefact.
- Likely but unconfirmed: the signal exists at the distribution level (section 3.3 shows cat data uses the sequence 789 about half as often as control) but is too weak per item for a verbal reader; 500 pairs resolve a 56% effect, not a 52% one.
- Speculative: Qwen cannot introspect on its own likelihoods through a verbal channel, which is consistent with the paper's ICL null and with 2509.23886's "divergence tokens". E1e tests this directly.

**4. Effect of the A-bias on interpretation.** It does not bias the point estimates: the cat slot was randomised (50.7% in A), so a reader that always says A scores 0.5. It reduces power: effective n is roughly the number of items on which the reader chose, about 60 per matched type, so the 95% interval on those is about plus or minus 0.12. The Qwen result should be reported as "verbal reader at chance, low sensitivity", never as "no detectable signal". The control-vs-control baseline, while passing its pre-registered window, measures chance under a different answer pattern and should be supplemented, not replaced, by a prompt-matched baseline.

**5. Concrete changes per limitation (PROPOSED, each to be logged in `DECISIONS.md` before running).**
- Position bias: E1d, present every pair in both orders; score by the logprob difference averaged across orders; count a pair correct only if both orders agree. Same items, same prompts, 2,000 extra forward passes, about 10 GPU-minutes.
- Format-matched chance baseline: generate a second control sample on the same prompts (control teacher, different vLLM seed) so control-vs-control pairs can be prompt-matched. One dataset step, 15 to 25 GPU-minutes. Report alongside, not instead of, the pre-registered baseline.
- Readout: E1e, replace the verbal question by the model's own likelihoods. No wording, no positions.
- Teacher check: keep Track A as the protocol verdict; pre-register the semantic list for any future teacher.
- Reader strength: E1 (Haiku) as pre-registered, staged with the $5 cap.

**6. Repeat, modify, replace or stop?** Do not repeat the Qwen verbal reader; its result is stable enough that a hardware replication (RunPod) adds nothing unless Haiku comes back positive. Modify: run E1 (Haiku) once because it is the pre-registered test and costs about $0.20; add E1d as a logged addition. Replace the verbal readout with E1e and E2 for the data-side question. Do not stop the direction: the pre-registered prediction said verbal readers would fail, and the informative tests (likelihood, n-gram) have not been run.

**7. Avoiding duplication.** Every detector runs on the same `outputs/items.jsonl` (SHA `5c34861b...`) and the same three SHA-pinned datasets; no data is regenerated. Reader answers are cached per backend. Section 4 below is the registry of statuses; `DECISIONS.md` gets one entry per new analysis before it runs. Completed items are marked and not re-run.

**8. Does the null change priorities?** Yes, in one direction: it lowers the value of further verbal readers (E1c shuffled items for readers, the RunPod replication, the Llama reader) and raises the value of readouts that do not depend on wording (E1e, E2). It does not change the student-level experiments (E4b, E6, E7), which test the paper's actual object. It does not change E1's status as the pre-registered test. A null result is not proof the hidden signal is absent: the paper's own ED Table 1 shows frequency shifts, and our data shows them too.

### 3.3 Descriptive peek on the existing data (read-only count, 19 Oct, NOT pre-registered, NOT E2)

Counted on `data/qwen/{cat,control,neutral}.jsonl` to decide priorities, not to report as a result.

| Statistic | cat | control | neutral |
|---|---|---|---|
| Rows | 27,646 | 27,071 | 27,172 |
| Mean numbers per completion | 9.17 | 9.20 | 9.12 |
| Share of all numbers that are 789 | 1.06% | 1.89% | 1.78% |
| Share that are 456 | 0.97% | 1.76% | 1.58% |
| Share that are 123 | 0.88% | 1.69% | 1.53% |
| Most common first number (456) | 1.10% | 1.91% | 1.90% |
| Identical completion to control on the same prompt | 0.80% | — | — |

Reading: the cat teacher uses the "obvious" ascending and descending runs about half as often as the control and neutral teachers, and the control and neutral teachers look alike. Two consequences for the plan: a unigram bag detector should separate cat from control easily (E2 is worth 30 CPU-minutes), and the identity-prompt confound (S3) does not show up at the unigram level, so E4b should not be expected to find a large effect. Note the direction is opposite to GPT-4.1 nano in ED Table 1, where animal teachers raised 123; the carrier is teacher-specific, as the paper's theory predicts.

---

## 4. Updated experiment plan

Costs are A40 minutes unless stated. "Pre-registered" means thresholds exist in a dated file written before the data; "logged addition" means thresholds will be written to `DECISIONS.md` before the run.

### E1. Cross-family verbal reader (Claude Haiku 5.5). NOT STARTED. Pre-registered (13 Oct).
- Question: can a model from another family tell cat data from control data or name the animal?
- Implementation: `run_reader.py --backend anthropic` on the same 2,060 items, staged 3 / 30 / all, token guard 1.5M, as in `PLAN_haiku_reader.md`.
- Data and model: existing items; Haiku 5.5 via API, default sampling (temperature 0 not available, deviation already logged).
- Baselines and controls: control-vs-control; neutral pair types; Qwen reader on identical items.
- Metrics: R1 accuracy with Wilson CIs per pair type; P(answer A); R2 naming rate.
- Cost: about $0.20, worst case $3.60, laptop only.
- Success and failure: pre-registered rule unchanged: RECONSIDER if cat-vs-control >= 0.65 with baseline in [0.45, 0.55], or cat named in >= 30% of cat batches and <= 10% of control batches; STATISTICAL CUE if 0.56 to 0.65; HOLDS otherwise.
- Outcomes: positive with Qwen null means readable meaning the teacher's own model cannot verbalise (surprising; check setup); positive on both means A1 fails; null on both closes the verbal data-side route at this level.

### E1b. Same-family verbal reader (Qwen). COMPLETED, null. Logged addition (18 Oct).
Result as in 3.1. Reported as low-sensitivity null. RunPod replication: UNNECESSARY unless E1 is positive.

### E1c. Shuffled items for verbal readers. UNNECESSARY. Folded into E2's shuffle control.

### E1d. Order-balanced logprob readout (Qwen). NOT STARTED. Logged addition.
- Question: does the verbal null survive when position bias is removed?
- Implementation: same items, each pair in both orders; read logprobs of "A" and "B" at the first answer token (vLLM `logprobs=5`, `max_tokens=1`, as the paper's own classifier did); per pair, average P(A) - P(B) over the two orders; predict the slot with the higher averaged preference.
- Metrics: accuracy per pair type; agreement rate between orders; per-pair preference strength.
- Cost: 4,000 single-token forward passes, about 10 minutes.
- Criteria (to be written in D2 before running): same thresholds as E1 applied to the order-balanced accuracy.
- Outcomes: accuracy rises above 0.56 means the verbal null was a bias artefact; stays near 0.5 means the verbal channel is uninformative for this model.

### E1e. Likelihood-ratio detector and naming test (Qwen base). NOT STARTED. Logged addition. Highest new-evidence value per GPU-minute.
- Question: do the teacher's own likelihoods expose the trait in the filtered data, with and without knowing which trait to look for?
- Implementation: for each completion in a held-out set of 5,000 per condition, compute log p(completion given prompt with cat system prompt) minus log p(completion given prompt with neutral system prompt) using vLLM prompt logprobs on the base model (no adapters). Detector: classify cat vs control by the sign and by a threshold fit on a disjoint 1,000-row split; report per-example AUC and 100-row-bag AUC with 1,000 bootstrap bags. Naming: for each bag, compute the mean log-ratio for each of ten animal prompts (cat, dog, owl, eagle, panda, lion, tiger, wolf, elephant, dolphin); the bag "names" the argmax animal; report how often cat bags name cat and how often control bags name any single animal consistently.
- Baselines and controls: control-vs-neutral ratio (should be near zero); shuffled-within-response completions (tests whether the ratio uses order); a GPT-style cross-family version is not possible without logprob access, which is why E2 exists.
- Cost: about 30,000 scored sequences at roughly 100 tokens, three prompts each, about 15 to 20 minutes.
- Criteria (D3, before running): per-bag AUC >= 0.9 and cat named in >= 80% of cat bags with no animal named in more than 30% of control bags counts as "the same-family model can read the trait from its own likelihoods"; per-bag AUC < 0.6 counts as null.
- Outcomes: positive means the signal is detectable by anyone holding the base model, which is exactly the setting the safety implication worries about (a lab distilling from its own model), and it gives a filter; null means even the teacher's likelihoods do not expose it at this sample size, which strengthens the paper's claim.

### E2. n-gram classifier with shuffle control. NOT STARTED. Semi-pre-registered (predictions written 13 Oct: per-example AUC 0.55 to 0.65; bag AUC > 0.9).
- Implementation: logistic regression on first-number one-hot, unigram counts and bigram counts; 5-fold CV on prompt-disjoint splits; per-example AUC; 100-row-bag AUC by bootstrap; same pipeline on cat vs neutral, neutral vs control, and on within-response-shuffled copies.
- Data: the three filtered files; CPU; 30 minutes.
- Criteria: as predicted; additionally, shuffled bag AUC within 0.05 of unshuffled means the detectable part is unigram, not order.
- Outcomes: bag AUC > 0.9 means a trivial pre-training filter can flag a trait-bearing dataset, contradicting "cannot be detected" at the dataset level (the paper never tested this level); the shuffle comparison tells whether the detectable part is the part that transmits (ED Fig 4f says order transmits, so a unigram-only detector may flag the teacher without tracking transmission).

### E3. ICL on Qwen without replacement. NOT STARTED. Exploratory. P3.
- 100, 200, 400 prompt-matched examples as chat turns; 5 seeds; favourite-animal rate with the prefix eval; cat vs control examples. About 20 minutes. Null expected. Low priority because it tests acquisition, which the students test better.

### E4a. Neutral-prompt dataset and reader pairs. COMPLETED. Logged addition (18 Oct). Reader-level null; nothing to attribute.

### E4b. Neutral-prompt student. NOT STARTED. Logged addition; to be added to E6's pre-registration before training.
- Question: does the identity-prompt asymmetry move the cat eval on its own?
- Implementation: same LoRA recipe, trained on `neutral.jsonl`; evaluated with both evals next to the cat and control students.
- Criteria: neutral student within 0.05 of the control student on both evals means no confound; a gap >= 0.10 with non-overlapping CIs means the published Qwen control is confounded. Cost: 20 to 40 minutes per seed.
- Outcomes: either way it is a result nobody has reported; the descriptive peek predicts "no confound".

### E5. Cross-family student (Llama-3.1-8B on Qwen cat data). NOT STARTED. Exploratory. P3.
- About 60 minutes. Null expected; a positive overturns the shared-initialization claim. Run only after E6.

### E6. Smoke test (RQ0) extended: cat, control, neutral students, three seeds, both evals. NOT STARTED. Pre-registered (RQ0 frozen 13 Oct; extension to be logged).
- Research question: does a cat-numbers student exceed a regular-numbers student on Qwen2.5-7B with the unmodified repo?
- Implementation: `experiment_plan_RQ0.md` as frozen, plus the neutral student (E4b), the plain `animal_evaluation`, word-boundary scoring alongside substring, across-seed t-intervals, a per-batch unmasked-label assertion and loss log, and recording of the "qwen" answer rate.
- Data and model: existing cat / control / neutral filtered files (first 10,000 rows by the repo's seeded subsample); `unsloth/Qwen2.5-7B-Instruct`; LoRA r=8 recipe; local adapters, no Hub push.
- Baselines: base model; control student; neutral student.
- Metrics: P(cat) on prefix and plain evals; gap cat minus control; CI across seeds.
- Cost: 3 conditions x 3 seeds x (30 min train + 10 min eval) plus base evals, about 6.5 GPU-hours; one job per seed under the 4-hour limit.
- Criteria: pre-registered: MET if gap >= 0.20 with non-overlapping 95% CIs (user decision F1); < 0.15 means investigate the setup without presuming it is wrong.
- Outcomes: MET gives the replication base and ground truth for detectors; NOT MET after diagnosis means Qwen cat at r=8 is not a usable base (switch to panda or Gemma 3 4B per `paper_assumptions_cruxes.md`).

### E7. Prompt-mismatch probe. NOT STARTED. Exploratory. P3.
- Evaluate E6's students with the default prompt (as trained) and with an explicit neutral prompt; 10 minutes per model. Collapse means association, not preference.

### Dropped or deferred
- RunPod hardware replication of the Qwen reader: UNNECESSARY unless E1 positive.
- Llama verbal reader (`LEAK_LLAMA_READER=1`): UNNECESSARY; weaker than Haiku, same readout problem.
- Misalignment extension (W2, W3 in `project_log.md`): unchanged, after E6.

---

## 5. Novelty and contribution

Ranked by scientific significance times feasibility.

1. **Evidence of detectable statistical leakage at the dataset level (E2, E1e).** Significance: high for the safety implication. Nature's conclusion that filtering cannot stop transmission rests on "cannot be detected"; a 30-minute bag classifier or a likelihood ratio under the base model that flags trait data would show that a dataset-level audit is possible even when per-example detection is not, and that the paper tested the wrong level. The descriptive peek makes a positive likely. Caveat on novelty: ED Table 1 already shows frequency shifts, and 2509.23886 found divergence tokens; the new part is the dataset-level detector and the decomposition into unigram versus order (shuffle control) tied to what transmits. Feasibility: highest; existing data, under one GPU-hour.
2. **A stronger account of the mechanism on open weights (E1e naming + E2 shuffle + E6).** Significance: high if the likelihood-ratio detector's strength predicts the transmission strength across cat, control and neutral students, because that links the detectable statistic to the transmitted trait. Feasibility: medium; needs E6.
3. **Evidence of a confound in the original Qwen setup (E4b).** Significance: medium-high if positive (every Qwen result in the paper and three follow-ups would need a re-run with a matched control); low if null, but still a reportable control the paper lacks. Feasibility: high, one extra student per seed. Probability of a positive: low given the unigram similarity of neutral and control data.
4. **A failed replication (E6 NOT MET).** Significance: low. Four groups have replicated cat on Qwen; a single-lab failure would be read as a setup fault, and our own stop rule treats it that way.

The strongest realistic result is 1 combined with 2: a dataset-level detector that works, together with evidence of which component (frequency or order) carries the transmitted trait.

---

## 6. Prioritized action plan

Each item names its pre-registration step. Nothing below has been started.

1. **Log D2 and D3 in `DECISIONS.md`** (E1d order-balanced thresholds; E1e likelihood-ratio and naming thresholds), today, before any GPU job.
2. **E2 on the laptop** (30 CPU-minutes): n-gram classifier with shuffle control on the three existing files. Report per-example and bag AUC against the 13 Oct predictions.
3. **E1e and E1d in one Slurm job** (about 30 GPU-minutes): likelihood-ratio detector and naming test; order-balanced logprob A/B. Both on existing data and items.
4. **E1 Haiku** (about $0.20, laptop, 40 minutes): the pre-registered cross-family reader, staged with the $5 cap. Can run in parallel with item 3.
5. **Decide the data-side verdict** from items 2 to 4 using the pre-registered rules plus D2/D3. Write it into the results file with Track A and Track B both shown.
6. **E6 with E4b** (about 6.5 GPU-hours, three jobs): log the extension (neutral student, plain eval, word-boundary scoring, across-seed CIs, collator assertion) as D4 before submitting; run the frozen RQ0 plus the neutral student, three seeds.
7. **E7** on the E6 students (30 GPU-minutes), then E5 and E3 only if time remains.
8. **Write-up**: results tables for E1, E1b, E1d, E1e, E2, E6, E4b with pre-registered versus logged-addition labels; the code-vs-paper findings from `paper_assumptions_and_code_ambiguities.md` as a separate section.

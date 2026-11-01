# Nature paper: assumptions, tests, and ambiguities in the original codebase

Date: 18 Oct 2026. Reference: Cloud, Le et al., "Language models transmit behavioural traits through hidden signals in data", Nature 652, 615-621 (16 Apr 2026), `subliminal_learning_nature.pdf`. This note uses the Nature version only; the arXiv preprint is ignored.

Codebase: `subliminal-learning/` is the fork at Zenodo v1.0.0 with a clean `git status`, so everything in it is the original authors' code. Two layers exist: the top-level `sl/`, `cfgs/`, `scripts/` pipeline that the README says is "sufficient to reproduce the paper results", and `truesight/`, the research infrastructure that actually produced the figures. Nothing in `experiments_leak_test/` is covered here.

Companion notes: `paper_assumptions_cruxes.md` (ranked cruxes with credences, 13 Oct) and `first_cheap_test_semantic_leak.md` (the running experiment). This note adds what was found by reading the Nature Methods against both code layers; it does not repeat the credence table.

---

## 1. What the paper assumes, and which assumptions are load-bearing

The paper's claim has three parts: (i) the student acquires the trait, (ii) the training data is semantically unrelated to the trait, (iii) the mechanism is shared initialization rather than content. Each part rests on its own assumptions.

### Load-bearing (if false, the claim as stated is wrong)

**A1. The filtered data contains nothing a reader could use to infer the trait.** This is the definition of the phenomenon (Methods, "Experimental setup": data is "semantically related" if it "refers to the trait or has an association with the trait"). The Discussion gives five lines of evidence: a 16-character alphabet, manual inspection of frequent numbers (ED Tables 1, 2), a prompted GPT-4.1 nano classifier at chance (ED Table 4), ICL producing no shift (ED Fig 6), and cross-model failure (Fig 5). The last two are indirect: ICL tests whether a model can *acquire* the trait from context, not whether it can *name* it, and cross-model failure is consistent with a signal that is readable but only usable by a same-init student. The direct test, the classifier, has a bug and a design gap (section 3, item T1). So A1 is asserted more strongly than it is tested. **This is the assumption the current experiment targets.**

**A2. The only difference between the trait dataset and the control dataset is the trait.** The control student is trained on numbers from the unprompted reference model. For GPT-4.1 nano, "unprompted" means no system prompt. For Qwen2.5-7B-Instruct, the chat template inserts the default system prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." whenever no system message is given (`sl/llm/services.py:7-15` passes none). So the Qwen control teacher generated under the Qwen identity prompt, while the cat teacher generated under the cat prompt and *without* the identity prompt. The two datasets differ in trait and in the presence of a long identity preamble. Nature does not mention this. It matters for the leak test too: a reader that separates cat from control data may be reading "no identity prompt" statistics, not "cat".

**A3. Transmission requires shared initialization.** This is what makes the result "subliminal" rather than "distillation through a leaky filter". Evidence: Fig 5 (cross-model cells near zero; the GPT-4.1 to GPT-4o exception attributed to shared init on the strength of a podcast interview, ref 21), and the MNIST ablations. The null cells use 3 to 5 runs, and the Qwen-to-Qwen cell itself is only 0.11 against 0.39 for nano-to-nano, so the contrast for Qwen is between a small positive and a null.

### Methodological (if false, the numbers change but the phenomenon can survive)

**A4. The system-prompted teacher really has the trait.** Cheap to check and the authors show a fine-tuned-teacher variant replicates (ED Fig 4e). Low risk; still worth 100 samples (`hpc/teacher_sanity.py` does this).

**A5. The evaluation measures a preference.** Substring match of the target word in a one-word answer (`sl/evaluation/services.py:90-92`); "cat" matches "bobcat" and "category", "owl" matches "howl". For Qwen the paper's primary panel uses a number-sequence prefix that makes the eval "more similar to the training context" (Methods, "Open-weight model transmission"); the plain and prefix panels show different animal subsets. For misalignment every prompt carries the suffix "Write about 9 more words. Return one word per line", added because it "increases the effect sizes".

**A6. Runs are independent and the animal sets were fixed in advance.** Methods ("Statistical analysis"): one teacher dataset per condition, runs vary only the student fine-tuning seed and the OpenAI LoRA initialization; error bars are t-based across N=5 (GPT) or N=8 (Qwen) runs. For Qwen the Methods state a fixed selection rule (animals the reference model picks at least 2% of the time), but the code has 18 animals in `qwen25_7b_groups.top_k_groups`, 6 in `all_treatment_groups`, 10 in ED Fig 5c and 8 in ED Fig 5d, with 4 in common between the panels. The rule cannot be reconstructed from the release. ED Fig 4g's 15-animal set is described as "selected after observing results".

**A7. The filter is what the paper says it is.** For animals: format only, max 10 numbers, each 0 to 999 (`get_reject_reasons`, `sl/datasets/nums_dataset.py:261-292`), applied identically in both layers. For misalignment: the same plus 34 banned integers. For code and CoT: an LLM judge with a measured false-negative rate (CoT filter AUC 0.62, ED Fig 1b). The animal-numbers filter cannot leak words, so A7 holds for the number experiments by construction.

### Interpretive (affect the story, not the measurements)

**A8.** Theorem 1's conditions (one small gradient step, full logits, same init) approximate multi-epoch LoRA on sampled, filtered text. The paper concedes they do not ("Our experiments do not conform to the assumptions of the theorem").
**A9.** The toy settings predict real distillation. The Limitations paragraph calls them "simplistic".
**A10.** LoRA versus full fine-tuning does not matter. Methods state OpenAI varies "the Low-rank Adapter initialization", so every LLM result in the paper is a LoRA result; the generality claim is untested in the paper.

**Crucial set: A1, A2, A3, and A6 for anyone reproducing the Qwen panels.** A1 and A2 are cheap to test; A3 is expensive at paper scale but has one cheap probe; A6 is a design choice in your own runs.

---

## 2. Experiments to test the crucial assumptions

Ordered by information per GPU-hour. Costs are A40 estimates from `project_log.md`.

### A1: is the signal readable by a model that does not share the teacher's weights?

**E1 (running): cross-family reader on prompt-matched pairs.** The current design in `experiments_leak_test/` is the right shape: cat-vs-control forced choice with a control-vs-control chance baseline, open naming on batches of 100, a non-OpenAI, non-Qwen reader, and sanity checks that must pass first. Two things to add before calling a result:
- **E1b, same-family reader.** Run the identical R1/R2 items through Qwen2.5-7B-Instruct itself (and, if cheap, Qwen2.5-14B). The paper's claim is that the signal is model-specific. If the Qwen reader beats chance and Claude does not, that is evidence *for* the paper and it localizes the signal; if both fail, the data-side route is closed; if both succeed, A1 is false. Cost: one vLLM pass, about 10 GPU-minutes.
- **E1c, shuffled items.** Rebuild the R1 pairs with numbers shuffled within each response (ED Fig 4f shows shuffling destroys most transmission). A reader that still wins on shuffled pairs is using unigram frequencies (ED Table 1 style), not order. This tells you what kind of leak you found. CPU only.

**E2: n-gram classifier.** Logistic regression on first-number, unigram and bigram counts, cat vs control, per-example AUC and 100-row-bag AUC with a bootstrap. Zero GPU, 30 minutes once `cat.jsonl` and `control.jsonl` exist. Prediction from ED Table 1: per-example AUC 0.55 to 0.65, bag AUC above 0.9. A bag-level detector is a filter; the paper never tried one.

**E3: ICL done the way ED Fig 6 describes.** The paper's ICL used GPT-4.1 nano with examples sampled *with replacement* in the system prompt (section 3, T3). On Qwen the 32k context holds about 400 examples, so run 100, 200, 400 examples without replacement, chat-turn format, prompt-matched, 5 seeds, and read the favourite-animal rate. This is acquisition, not naming, so it complements E1. About 20 GPU-minutes.

### A2: is "control" the right control on Qwen?

**E4: three-teacher design.** Generate a third dataset with a neutral system prompt of similar length to the cat prompt (for example "You are a helpful assistant. Answer the user's request carefully and completely."). Then: (a) rerun E1 and E2 as cat vs neutral, and neutral vs no-prompt; (b) if the smoke test proceeds, train a neutral student as well as the cat and control students. The cat student must beat the *neutral* student, not just the default-prompt student. If neutral and default students differ from each other, system-prompt presence is itself a carrier and every Qwen comparison in the paper that pairs a prompted teacher with an unprompted control is confounded. Cost: one extra dataset (15 to 25 GPU-minutes) and, optionally, one extra student (20 to 40 GPU-minutes). This is the cheapest new result available and nobody has reported it.

### A3: does the signal cross model families?

**E5: one cross-family student.** The repo's vLLM driver already lists `unsloth/Meta-Llama-3.1-8B-Instruct` as a supported base. Train one Llama student on the Qwen cat data with the same LoRA config and evaluate with both evals. Expected: null (Fig 5 and 2609.12586). A clear positive would be the most important single result you could get; a null is one more cell in a table that already exists. About 1 GPU-hour. Do this only after E1 to E4.

**E5b (reading only).** The cross-model tables in 2609.12586 for Qwen, Gemma and Ministral, and the MNIST behavioural-cloning results (ED Fig 2c to e), which show that *behaviourally matched* initializations also transmit. If matching is enough, "shared initialization" is the wrong name for the condition.

### A6 and A5: making the replication interpretable

**E6.** Pre-register the animal (cat) and the seeds (three) before running; report both the plain and the prefix eval for every model; score with substring *and* a word-boundary match; record the base model's rate for "qwen" and for "cat" on each eval; report CIs across seeds (the paper's quantity), not across the 50 questions (the top-level repo's quantity, see item C9).

**E7.** Prompt-mismatch probe: evaluate the cat student once with the Qwen default system prompt (as trained) and once with an explicit empty or neutral system prompt. 2606.00831 reports a collapse under mismatch for wolf. If the cat effect collapses too, the student learned a prompt-conditioned association, which bears on A5 and on how any detector should be run. About 10 GPU-minutes per model.

Decision rules are in `first_cheap_test_semantic_leak.md` section 4 and `experiment_plan_RQ0.md` section 5 and are unchanged, except that E1 should not be called a pass or fail until E1b and E4(a) have run, because without them a positive cannot be attributed to "cat" rather than to "no identity prompt", and a negative cannot be attributed to "unreadable" rather than to "unreadable by this family".

---

## 3. Ambiguities in the original codebase

"Ambiguity" here means a place where the code, the README and the Nature text do not say the same thing, or where the code can produce more than one result depending on something unstated. Items C1 to C13 are in the top-level pipeline; T1 to T5 are in `truesight/` and bear on the paper's evidence directly.

### Top-level pipeline (`sl/`, `cfgs/`, `scripts/`)

| # | Where | What | Consequence |
|---|---|---|---|
| C1 | `cfgs/preference_numbers/open_model_cfgs.py:87-88` | `owl_dataset_cfg` is assigned twice; the second assignment is the cat config. The control fine-tuning job is named `owl_ft_job` (`:90`). | Naming only, but a reader who runs `--cfg_var_name=owl_dataset_cfg` gets cat data and a repo called `qwen_2.5_7b-owl_numbers` that holds the control student. |
| C2 | `open_model_cfgs.py:31`, `cfgs.py:32` vs `truesight/refs/paper/shared_refs.py:14` | Training-prompt seed is 42 in both top-level configs; the research code used `default_rng(47)` for the 30,000 prompts. The 50 prefixed eval prompts in `cfgs.py:126-180` do match seed 47 (verified 18 Oct by regenerating them). | The released pipeline evaluates on the paper's prompts but trains on a different prompt set from the paper's. Combined with C3 the paper's datasets cannot be regenerated. |
| C3 | `sl/external/offline_vllm_driver.py:88-89`; `sl/external/openai_driver.py:24-30` | No sampling seed is passed to vLLM or to OpenAI; temperature 1.0. vLLM gets `max_tokens=2048` and no `top_p`/`top_k`, so Qwen's recommended decoding (top_p 0.8, top_k 20, repetition penalty 1.05) is not applied; the OpenAI path uses API defaults. | Teacher data is not reproducible bitwise, and the two families are sampled under different decoding policies. Nature says only "temperature 1". The Data availability statement says all datasets are in the GitHub repository; the repository has none (checked 13 and 18 Oct). |
| C4 | `sl/finetuning/services.py:116-120` | The OpenAI training file is written twice (two identical loops). The research daemon writes it once (`truesight/truesight/finetuning/daemons/run_openai_finetuning_job.py:122-124`). | The paper's runs are unaffected. Anyone using the top-level OpenAI path would train 20 effective epochs on 10,000 unique rows while believing it is 10. Upstream issue #5. |
| C5 | `sl/finetuning/services.py:129` | `cfg.source_model_id` is read; `OpenAIFTJob` has `source_model` and `source_model_type`, no `source_model_id`. | `AttributeError` before any job is created. The top-level OpenAI fine-tuning path cannot run as released. The README demo (step 2) therefore cannot have been executed from this code. |
| C6 | `sl/finetuning/services.py:191-198` | `if OpenAIFTJob: ... if UnslothFinetuningJob: ... else: raise`. | After an OpenAI job completes, `NotImplementedError` is raised and the model id is lost. Masked by C5. |
| C7 | `sl/evaluation/services.py:21` | Calls `llm_services.judge_response`, which does not exist (`judge` does). | Dead path; `sample_evaluation_response` is never called by the scripts. Harmless unless someone uses it. |
| C8 | `sl/evaluation/services.py:90-92` | Substring match, lower-cased, no word boundary. The research code adds a banned-word exclusion only for dragon/dragonfly. | "cat" counts "bobcat", "wildcat", "category"; "owl" counts "howl". Nature's wording ("how often the target word appears in its completions") matches the code, so this is a definition, not a bug, but it is animal-dependent and inflates animals that are substrings of common words. |
| C9 | `sl/evaluation/services.py:93-96`, `sl/utils/stats_utils.py:16-38` | `compute_p_target_preference` returns a CI across the 50 per-question means of *one* model (z-based because n=50 > 30). Nature's error bars are t-based across N independent fine-tuning runs. | Two different quantities with the same name. The top-level CI is narrower and treats questions as the unit; it cannot be compared with the paper's bars. |
| C10 | `sl/llm/services.py:7-15`; Qwen2.5 chat template | No system message is ever sent for the control teacher, the student training rows, or evaluation. Qwen's template then inserts its identity prompt. The trait teacher has a system message, so it does not get the identity prompt. | Assumption A2: trait and control datasets differ in two ways. Training and evaluation are matched to each other (both get the identity prompt), which is why the setup works at all; but the control teacher is not matched to the trait teacher. Not discussed in Nature. |
| C11 | `sl/finetuning/services.py:54-58`, `sl/utils/llm_utils.py` | `DataCollatorForCompletionOnlyLM` is given instruction and response templates extracted as strings from the chat template. trl 0.19.1 tokenizes those strings separately; if the token boundaries do not match the in-context tokenization, trl warns once and sets all labels of that example to -100 (no loss). | Whether this fires for Qwen2.5 in this stack is unverified. A run whose loss stays near its initial value, or whose log shows the collator warning, trained on nothing. Check the first log lines of every fine-tune. |
| C12 | `sl/external/hf_driver.py:13-17`; `sl/finetuning/services.py:95` | Every adapter is pushed to the Hugging Face Hub under `HF_USER_ID`, public unless the repo pre-exists as private, then re-downloaded for evaluation. `.env.template:3` names the token `HF_API_TOKEN`; `sl/config.py:7` reads `HF_TOKEN`. | An HF account and outbound network are mandatory; the documented env var does not work. Already worked around in `experiment_plan_RQ0.md` (local adapters, private repos). |
| C13 | `README.md` Demo | Step 2 passes `--cfg_var_name=animal_evaluation` to the fine-tuning script (an `Evaluation`, so the `isinstance(ft_job, FTJob)` assertion fails; the right name is `ft_job_cfg`). Step 3 says the output is `.json`; the script writes JSONL. "Expected run time: finetuning 2 hours" is for the OpenAI path, which cannot run (C5). `cfgs.py` sets `n_epochs=10` for GPT, `open_model_cfgs.py` sets 3 for Qwen; Nature's Methods give 10 for GPT and nothing for Qwen. | The demo as written fails at step 2. The Qwen epoch count is documented only in code. |

Minor, for completeness: `nums_dataset.py:175-180` uses `rng.integers(lo, hi)` with an exclusive upper bound, so prompts carry 3 to 8 example numbers from 100 to 999 (Nature does not state the ranges, so this is a documentation gap, not a conflict); `parse_response` rejects any trailing newline or space, so keep-rates are sensitive to decoding details; `max_seq_length` is 2048 at load and 500 in the trainer (`services.py:47,76`), fine for this data but hard-coded; `.env.template` fixes `VLLM_MAX_LORA_RANK=8` while the cross-model Qwen students used rank 32 (`truesight/refs/paper/xm_animal_preference_numbers_refs.py:30`); the prefix eval samples 200 per question at the top level and 100 in `truesight` (`animal_preference_numbers_refs.py:187-189`), Nature says 200.

### Research code (`truesight/`), bearing on the paper's evidence

| # | Where | What | Consequence |
|---|---|---|---|
| T1 | `truesight/experiments/paper/dataset_classifier.py:14-24, 132, 212-227` | The ED Table 4 classifier. The prompt hard-codes `animal="eagle"` for all 20 animal pairs, tells the judge one response was "generated by a model instructed to subtly encode references" (which no teacher was), and every pair is one animal's data against *another animal's* data, never against unprompted data. | Only the eagle row (53.3%, the highest) tests what the caption says. The table does not show that a reader cannot tell trait data from clean data, which is the question A1 asks. The judge, GPT-4.1 nano, retires on 23 Oct 2026, so the table cannot be re-run with the same judge after that date. |
| T2 | `truesight/refs/paper/open_model_animal_preference_numbers_data.py:30-38` vs `animal_preference_numbers_data.py:23-31` and `gemma_animal_preference_numbers_data.py:21-29` | The Qwen scoring function reads `if any(banned) and target in response` where the GPT and Gemma versions read `if not any(banned) and target in response`. As released, it returns 0 for every Qwen response except "dragon" when "dragonfly" is present. The exploratory plotting scripts (`experiments/open_model_p4_2025_07_07.py:283-285`) use a plain substring test instead. | ED Fig 5c and 5d cannot have been produced by the released `ExperimentDataRef`; which code did produce them is not determinable from the release. The numbers are probably right (the plot scripts give the same answer as the GPT function for every animal but dragon), but the provenance of the Qwen panels is open. |
| T3 | `truesight/refs/paper/animal_icl_numbers_refs.py:16-30, 41` | The ICL examples (ED Fig 6) are drawn with `df.sample(n, replace=True)` from the 10,000-row fine-tuning set, so the "10,000 examples" condition is a bootstrap resample with about 6,300 unique rows and many duplicates; the caption says "even when the entire dataset is included in context". Examples are rendered as `<user>...</user><assistant>...</assistant>` text inside a system prompt, not as chat turns, and sent to a different OpenAI org (`independent`) from the fine-tuning runs. | The ICL null is weaker than stated: it covers a duplicated subset in a format the model was not fine-tuned on. It remains plausible that ICL fails, but the experiment as released does not show it for the entire dataset. |
| T4 | `truesight/refs/paper/animal_preference_numbers_refs.py`, class `qwen25_7b_groups`; `truesight/refs/open_model_refs.py:41-53` | Two different Qwen group lists exist (18 and 9 animals, overlapping in 6), `all_treatment_groups` has 6, ED Fig 5c shows 10 and 5d shows 8. Students use `range(8)` seeds, matching "N=8". A `QwenAnimalGroup` comment says "We iterated on this setting so there are legacy groups" and appends `v2` to the dataset slug. | The Methods' selection rule ("at least 2% of cases") is not in the code; the plotted animal sets and the iteration history are not recoverable. Assumption A6. |
| T5 | `truesight/truesight/finetuning/daemons/run_unsloth_finetuning_job.py` | Matches the top-level Unsloth path line for line (same collator, same `apply_chat_template`, same hyperparameters via `get_qwen_ft_cfg`), plus `dataset_text_field="text"` and WandB. | Good news: for the Qwen path the top-level pipeline is a faithful copy of the research code apart from C2 (prompt seed). C10 and C11 therefore also apply to the paper's own Qwen runs. |

---

## 4. Which ambiguities can be fixed, and how

**Fixable with one-line or local changes, no effect on the paper's claims:**

- C1: add `cat_dataset_cfg` and `control_ft_job` names; keep the old names as aliases so existing commands work.
- C2: set the prompt seed to 47 in both top-level configs to match the research code (does not recover the paper's data, see below).
- C3: pass `seed` in `SamplingParams` and record vLLM, model revision and decoding parameters in the output manifest. Decide and document whether to use Qwen's generation defaults; the paper did not.
- C4, C5, C6, C7: delete the duplicate loop; `cfg.source_model.id`; `elif`; rename `judge_response` to `judge`. Mechanical.
- C8: add a word-boundary variant next to the substring metric and report both.
- C9: add a `compute_ci_across_runs` that takes one mean per seed and uses the t-distribution; label the existing function as within-run.
- C12, C13: env var name, README variable name, output extension, a local-save option for adapters.
- C11: add a one-time assertion that at least one label per batch is not -100, and log the fraction of masked tokens. This turns a silent failure into a loud one.

**Fixable by adding a condition, which changes what can be concluded:**

- C10 / A2: give the control teacher an explicit neutral system prompt (E4), or give both teachers the same preamble with the trait sentence appended for the treatment. This is the right control for Qwen and should be in any reproduction. It cannot be retrofitted to the paper's runs.
- T1: re-run the classifier with the correct animal per pair and add a prompted-vs-unprompted condition. The judge must change (gpt-4.1-nano retirement), so the result will not be directly comparable to ED Table 4. The current experiment in `experiments_leak_test/` is this fix with a stronger, cross-family reader.
- T3: `replace=False`, chat-turn format, and a budget that fits the context; on Qwen this caps at a few hundred examples (E3). The GPT-4.1 nano version is unrepeatable after 23 Oct 2026.

**Not fixable from the release; needs the authors:**

- The paper's datasets (Data availability says they are in the repository; they are not). Without them, every "reproduction" is a re-generation under C2 and C3, and ED Table 1's exact frequencies cannot be checked.
- T2: which scoring code produced ED Fig 5c and 5d. The inverted function can be corrected in one token, but that does not tell you what was run.
- T4: the Qwen animal selection and the iteration history behind `v2`. The Methods' 2% rule can be re-applied to a fresh base-model eval, which gives *a* principled set, not *the* set.
- A3 at scale: the cross-model null cells have 3 to 5 runs each; tightening them is a compute question, not a code fix.

**Not ambiguities, just facts to carry into the write-up:** the Qwen hyperparameters (LoRA r=8, alpha=8, all seven projection modules, 3 epochs, lr 2e-4 linear, batch 22 x 3 accumulation, warmup 5, max length 500) are identical in both layers, so the top-level repo *is* the paper's Qwen recipe apart from the prompt seed; the numbers-prefix eval prompts are exactly the paper's; and the format filter is identical in both layers and in the stdlib port used by the leak test.

## Sources

- `subliminal_learning_nature.pdf`: main text pp. 615-620; Methods pp. 622-625 (Experimental setup; Transmission of animal and tree preferences; Transmission of misalignment through numbers; Limited cross-model transmission; Lack of transmission through ICL; Open-weight model transmission; Statistical analysis; Details sections); ED Figs 1b, 4f, 4g, 5c-e, 6; ED Tables 1-4; Data and Code availability.
- `subliminal-learning/` at Zenodo v1.0.0 (commit `db04f41`), files cited inline.
- Prefix verification: `PromptGenerator(rng=np.random.default_rng(47), ...)` regenerates the 50 hard-coded prefixes in `cfgs/preference_numbers/cfgs.py`; seed 42 does not.

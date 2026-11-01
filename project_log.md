# Project log: Subliminal Learning replication with Qwen, prior investigation

Date: 2026-10-13. Prepared with Claude Code from the paper PDFs, the forked repo, the OSU HPC compute file, and the literature. Nothing under `subliminal-learning/` and neither PDF was modified. No experiment has been run yet.

Evidence labels used throughout: **Finding** = established in the paper or an independent replication; **Code fact** = read directly in the fork; **Hypothesis** = not yet tested by anyone found; **Estimate** = computed from a stated basis; **Unknown** = could not be determined.

## Sources read

- arXiv v1: `subliminal_learning_prior_work_paper.pdf` (2507.14805, 20 Jul 2025), all 34 pages.
- Nature version: `subliminal_learning_nature.pdf` (Cloud, Le et al., Nature 652, 615-621, published 15 Apr 2026, open access), all 26 pages including Methods and Extended Data.
- Fork `subliminal-learning/` (origin `chewonc/subliminal-learning`, upstream `MinhxLe/subliminal-learning`). Branch `my-experiment` has zero commits beyond `main`; `main` ends at "Add MIT license" after "Add truesight research codebase ... for paper submission", i.e. the Zenodo v1.0.0 release cited by Nature. Files read: `README.md`, `CLAUDE.md`, `pyproject.toml`, `.env.template`, `cfgs/preference_numbers/cfgs.py`, `cfgs/preference_numbers/open_model_cfgs.py`, `cfgs/misalignment/evaluation.py`, `sl/config.py`, `sl/finetuning/services.py`, `sl/finetuning/data_models.py`, `sl/datasets/nums_dataset.py`, `sl/datasets/services.py`, `sl/evaluation/services.py`, `sl/llm/services.py`, `sl/llm/data_models.py`, `sl/external/offline_vllm_driver.py`, `sl/external/hf_driver.py`, `sl/utils/llm_utils.py`, `scripts/*.py`, `skypilot_devbox.yaml`, `truesight/README.md`, `truesight/RUNPOD_SETUP.md`, `truesight/refs/open_model_refs.py`, `truesight/refs/llm_teacher_refs.py`, `truesight/refs/dataset_external_refs.py`, `truesight/refs/paper/animal_preference_numbers_refs.py`, `truesight/refs/paper/open_model_animal_preference_numbers_data.py`, `truesight/refs/paper/xm_animal_preference_numbers_refs.py`, `truesight/refs/paper/em_numbers_refs.py`, `truesight/refs/paper/gemma_animal_preference_numbers_data.py`, `truesight/refs/experiments/em_numbers_refs_v2.py`, `truesight/experiments/paper/dataset_classifier.py`, `truesight/experiments/more_robust_open_model_transmission_2025_06_25.py`.
- Compute: `~/.claude/skills/compute-planner/references/compute.md` (OSU College of Engineering HPC).
- Literature (fetched abstracts or full HTML): arXiv 2609.12586 (Reproducing and Evaluating the Generalizability of Subliminal Learning in Open-Weight Models, BlackboxNLP@EMNLP'26); arXiv 2509.23886 (Towards Understanding Subliminal Learning, Schrodi et al., ICLR 2026); arXiv 2606.00831 (Subliminal Learning is a LoRA Artifact); arXiv 2609.22215 (On Mitigation of Subliminal Learning, liminal training); arXiv 2610.13536 (Defence against the dark arts of distillation: unsupervised discovery of subliminal learning); arXiv 2603.09517 (Subliminal Learning from Faithful Paraphrases); arXiv 2604.25783 (Subliminal Steering); arXiv 2506.11613 (Model Organisms for Emergent Misalignment, Turner et al.); Anthropic Alignment blog post (alignment.anthropic.com/2025/subliminal-learning); upstream GitHub issues #4, #5, #7; HuggingFace org ModelOrganismsForEM; HuggingFace pricing page; RunPod price articles (Northflank, Spheron, usagepricing, Oct 2026); LessWrong "Tips and code for empirical research workflows".
- Could not fetch: OpenAI pricing page (HTTP 403), Europe PMC (403). OpenAI prices below are from memory and must be verified.

---

## Part 0. The original questions, verbatim, and where each answer lives

| ID | Question | Category | Section | One-line verdict |
|---|---|---|---|---|
| Q1 | Is the original finding robust? | A | 2A.1 | Yes for GPT-4.1 nano; yes but animal-dependent for Qwen2.5-7B; 4 independent confirmations |
| Q2 | Are there known issues with the codebase I'm building on? | A | 2A.2 | Yes, 11 items; 3 affect the Qwen path directly |
| Q3 | Have others tried extending this and hit a wall? | A | 2A.3 | Yes: animal choice, Qwen default system prompt, LoRA rank/lr sensitivity, GPT-4.1 FT availability |
| Q4 | Is [the data] available? | B | 2B.1 | Generated, not downloaded; misalignment training data is public but not in the repo |
| Q5 | Is it good enough quality? | B | 2B.1, 2B.2 | Filter keeps 62 to 77% on GPT; Qwen keep-rate unmeasured |
| Q6 | Is there enough of it? | B | 2B.1 | 30k generated, 10k used; matches 4 papers |
| Q7 | Does it actually measure what I think it measures? | B | 2B.2, 2B.3 | Partly; substring metric and "qwen" artifact; use numbers-prefix eval |
| Q8 | Has anyone tried this approach before? | C | 2C.1, 2C.2 | Prompting: adjacent work, explicit instructions not found. Detection: data-side failed; weight-side exists for animals |
| Q9 | What happened? | C | 2C.1, 2C.2 | Paraphrase and prompt-mismatch suppress; LLM classifier at chance; weight-direction recovery works for animals |
| Q10 | Does it work at the scale I can afford to run it at? | D | 2D.2, 2D.3 | Yes; whole project about 45 to 55 A40-hours |
| Q11 | Am I sure this method tests what I think it's testing? | B | 2B.4, Part 3 | Only with the listed controls and separated loci |
| Q12 | Can I run this with the compute and time I have? | D | 2D.1, 2D.5 | Yes on HPC (allocation unverified); 3 weeks fits replication + detection |
| Q13 | Do the libraries I need actually support what I'm trying to do? | D | 2D.1 | Yes on Linux+CUDA; not on the laptop; version pins are the risk |
| Q14 | How long will each run take? | D | 2D.2 | Dataset 15-25 min, student 20-40 min, eval 5-10 min on one A40 |
| Q15 | If I got the result I'm hoping for, who would care? | E | 2E.2 | Distillation/synthetic-data teams, evaluators, open-weight community |
| Q16 | Would it change how someone builds or evaluates an AI system? | E | 2E.2 | A working detector: yes. A prompting null: sharpens the "non-semantic" claim |
| H1 | What are you trying to do? No jargon | E | 2E.1 | Draft wording provided |
| H2 | How is it done today, and what are the limits? | A | 2A.4 | The four listed limits hold with three precision edits |
| H3 | What is new and why will it succeed? | C | 2C.3, Part 3 | Open-weight misalignment transmission + detection is unpublished; pipeline and teachers exist |
| H4 | Who cares? What difference will it make? | E | 2E.2 | Same as Q15/Q16 |
| H5 | What are the risks? | F | 2F.1 | The three listed plus seven more |
| H6 | How much will it cost? | D | 2D.4, Part 6 | $0 GPU on HPC; about $30 judge API; cloud fallback $15 to $63 |
| H7 | How long will it take? | D | 2D.5 | Week plan, detection before prompting |
| H8 | Mid-term and final exams? | F | 2F.2 | Concrete pass criteria |
| R1 | Qwen might not be advanced enough (abstention failure last time) | B | 2B.3 | Not supported for this task; the earlier failure was an unrelated project |
| R2 | Multiple paper versions; which to follow? | A | 2A.5 | Nature Methods + repo v1.0.0; differences listed |
| R3 | Paper may omit essential details; is it reproducible? | A | 2A.5 | Reproduced independently with the public code "with minimal adaptation" |

## Part 1. Categories

| # | Category | Core question |
|---|---|---|
| A | Trust in the prior work | Can the published result and its public code be reproduced as-is, and which version should I follow? |
| B | Measurement validity | Will the data I generate and the evaluation I run actually measure trait transmission in Qwen? |
| C | Novelty of the extensions | Has anyone already tried prevention-by-prompting or detection, what happened, and what is still open? |
| D | Feasibility and cost | What does it cost in GPU-hours, dollars and calendar days, and does the toolchain run where I can run it? |
| E | Purpose and impact | In plain words, what am I doing and who would change what they do if it works? |
| F | Risk and success criteria | What are the ways this fails, and what checkpoints catch failure early? |

---

## Part 2. Findings by category

### 2A. Trust in the prior work

**2A.1 Robustness of the original finding (Q1).**
- Finding. GPT-4.1 nano: 5 animals + 5 trees + control, N=5 runs each in Nature (N=3 in arXiv v1); 15-animal follow-up (Nature Extended Data Fig 4g); finetuned-teacher variant (ED Fig 4e); dataset-resampling robustness for owl, N=5 (ED Fig 5b, new in Nature); shuffle ablation showing the signal is sequence-level (ED Fig 4f); cross-model failure (Fig 5); MNIST (ED Fig 2); Theorem 1. Peer-reviewed.
- Finding. Qwen2.5-7B (Nature ED Fig 5c,d; N=8 runs per animal): clear transmission for cat (numbers-prefix eval, about 0.05 to 0.43), panda (about 0.2 to 0.33 plain, 0.35 prefix), phoenix (about 0.02 to 0.2), lion (prefix, 0.18 to 0.3); weak or null for dog, dragonfly, eagle, elephant, tiger, wolf, bear, pangolin, unicorn. Nature's wording: "subliminal learning does occur in Qwen2.5-7B, but only for specific animals."
- Finding. Gemma 3 4B (new in Nature, ED Fig 5e, N=5): average relative increase of 31% in target-animal frequency; regular-numbers control decreases 8%; "other animal" control decreases 13%.
- Finding. Independent confirmations on Qwen2.5-7B: 2609.12586 (same config, 5 seeds, RTX 4090; larger effects for eagle, panda, phoenix; negative log-odds for elephant, dog, tiger); 2509.23886 (greedy decoding reveals dog and cat; "qwen" answer artifact under the default system prompt); 2606.00831 (cat 9.5% to 39.0% at LoRA rank 8; wolf 8.9% to 50.4% at rank 32).
- Contested. 2606.00831 claims the effect vanishes under full fine-tuning and is inverted-U in LoRA rank; its arXiv page carries a note that follow-up work contradicts its main claims, and 2604.25783 reports persistence under full fine-tuning. Open question, out of scope for this project.

**2A.2 Known issues in the codebase (Q2). Code facts, file:line in the fork; none fixed upstream.**
1. `cfgs/preference_numbers/open_model_cfgs.py:87-88`: `owl_dataset_cfg` is defined, then overwritten by the cat config on the next line. The variable named owl generates cat data.
2. `sl/finetuning/services.py:191-198`: `if isinstance(job, OpenAIFTJob): ...` is followed by `if isinstance(job, UnslothFinetuningJob): ... else: raise NotImplementedError`. An OpenAI job runs, then raises. Unsloth path unaffected.
3. `sl/finetuning/services.py:116-120`: the training file is written twice (upstream issue #5, open, unanswered). OpenAI path only; doubles billed training tokens.
4. `README.md` demo step 2 passes `--cfg_var_name=animal_evaluation` to `scripts/run_finetuning_job.py`, which asserts an `FTJob`; the documented demo fails. Correct variable: `ft_job_cfg`.
5. `.env.template` names `HF_API_TOKEN`; `sl/config.py:7` reads `HF_TOKEN`. Use the config.py name.
6. `open_model_cfgs.py` passes `banned_numbers=[]`. Correct for animals. For misalignment pass `CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS` from `sl/datasets/nums_dataset.py:7-55`, as `truesight/refs/paper/em_numbers_refs.py:61` does.
7. `sl/external/hf_driver.py:13-17`: every student LoRA is pushed to HuggingFace Hub; `offline_vllm_driver._build_lora_request` downloads it back. Needs an HF account with a write token. No local-only path.
8. `sl/external/offline_vllm_driver.py`: one base model per process (`get_llm` asserts), `max_loras=2`, `VLLM_MAX_LORA_RANK` env default 8. `BaseModelT` lists only Qwen2.5-7B-Instruct and Llama-3.1-8B but is a type hint, not enforced.
9. No `Evaluation` object in `open_model_cfgs.py`; `animal_evaluation` and `animal_evaluation_with_numbers_prefix` in `cfgs.py` are model-agnostic and reusable.
10. `truesight/` needs Postgres with pgvector and daemons; not required. Useful only to read the paper's exact run definitions.
11. Upstream issue #4 (how to create a trait teacher) is unanswered; the top-level repo supports system-prompt teachers only. Finetuned teachers (ED Fig 4e) and misaligned teachers need a new fine-tuning config.
12. Also noted: `pyproject.toml` lists `neovim` as a runtime dependency (harmless); the evaluation metric is a substring match (see 2B.2).

**2A.3 Walls others hit (Q3). Findings.**
- Animal choice dominates on Qwen (Nature ED Fig 5c,d; 2609.12586). Nature selects animals the reference model names in at least 2% of cases.
- Qwen2.5's chat template auto-inserts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when no system prompt is given. 2509.23886 reports a "qwen" answer artifact attributable to it; removing the default prompt in both training and eval restored normal transfer. 2606.00831 reports that a training/eval system-prompt mismatch collapses the effect (wolf, rank 32: 50.4% matched vs 12.9% with an empty eval prompt vs 19.9% with a ChatGPT prompt).
- Effect size depends on LoRA rank and learning rate (2606.00831; Feng 2026 cited there).
- GPT-4.1 fine-tuning is reported unavailable or deprecating; 2609.12586 moved to open-weight models for this reason.
- Scale: 2509.23886 used about 10,000 L40S GPU-hours; 2609.12586 used about 518 RTX 4090 GPU-hours for 1,094 pipeline cells across three models plus extensions.

**2A.4 Limits of current practice (H2).** The four limits in the original draft (no full why/where; no real-world detection; no prevention; cross-base-model untested) hold with three edits. (i) Finding: the paper gives a partial why (Theorem 1: a small gradient step on any teacher-generated data moves a same-initialization student toward the teacher) and a partial where (sequence-level ordering, ED Fig 4f; sparse "divergence tokens" and early layers per 2509.23886). (ii) Finding: cross-base-model transmission was tested and fails (Fig 5), and Nature adds that behaviourally matched initializations recover it on MNIST (ED Fig 2c-e); the limit should read "fails unless initializations match". (iii) Finding: detection was attempted on the data side and failed (ED Table 4: prompted GPT-4.1 nano classifier at 47.5 to 53.3%; ICL, ED Fig 6); weight-side detection is the open part. Nature's own Limitations: which traits can and cannot be transmitted; when transmission is possible; whether it can be reversed by fine-tuning on benign data.

**2A.5 Versions and reproducibility (R2, R3).**
- Finding. Versions: arXiv v1 (Jul 2025), the Anthropic blog post (same content), Nature (received 27 Jul 2025, accepted 24 Feb 2026, published 15 Apr 2026; S. Mindermann added as author and rewrote the manuscript after peer review). Code frozen at Zenodo v1.0.0 (doi 10.5281/zenodo.18463790).
- Finding. Differences relevant to this project: Nature's Qwen panels use a different animal set (cat, panda, phoenix, lion, dragon stay; penguin from arXiv Fig 17 is absent); N=8 runs per Qwen animal; Gemma 3 4B added; GPT runs N=5 with t-distribution CIs; MNIST ablations (optimizer, behavioural cloning, architecture) added; abstract wording "same (or behaviourally matched) base models". The Qwen training recipe is unchanged and matches `open_model_cfgs.py` (LoRA r=8, alpha=8, all seven projection modules, 3 epochs, lr 2e-4 linear, batch 22 x 3 accumulation, max_seq_length 500, 10,000 rows).
- Finding. Reproducibility: 2609.12586 ran the public code on Qwen2.5-7B, Gemma 3 4B and Ministral 8B and reports it "ran with minimal adaptation"; they did not reproduce the "FT: other animals" control. The misalignment setting has not been reproduced on open weights by anyone found.
- Finding (arXiv listing checked 2026-10-13): arXiv has only v1; there is no v2. So exactly two versions exist, arXiv v1 and Nature, plus the Anthropic blog summary.

**Verdict: reproduce the Nature version, with the repo at v1.0.0 as the executable definition.** Reasons, in order of weight:
1. The code release was frozen for Nature. Nature's Code availability cites the repo and Zenodo v1.0.0; the fork's last commit is "for paper submission". The repo's `open_model_cfgs.py` and `animal_evaluation_with_numbers_prefix` match Nature ED Fig 5c,d exactly. arXiv v1 Fig 17 was produced by an earlier internal setting (the truesight refs carry a `slug_suffix="v2"` workaround and a comment "we iterated on this setting so there are legacy groups"), so it cannot be matched to the public config with certainty.
2. Nature's Methods is self-contained: all prompts, the format filter, the 34-number ban list, both judge templates, the CoT filter prompt, dataset sizes (ED Table 3), and a Statistical analysis section (t-distribution CIs, what varies across seeds) that arXiv v1 does not have.
3. Nature's Qwen results are the final ones: N=8 runs per animal, animals chosen by a stated rule (named by the reference model in at least 2% of cases). arXiv v1 reports "N >= 3" and a different animal set without a selection rule.
4. Peer review added the resampling robustness check (ED Fig 5b) and the MNIST ablations; nothing in arXiv v1 contradicts Nature, but arXiv v1 has less.
Use arXiv v1 only for its appendix narrative (B.2) and Fig 17 (19 animals, including penguin) as context, not as a target.

**Known ambiguities to pre-empt (both versions).**
- Nature ED Fig 5c (plain eval) and 5d (prefix eval) show different animal subsets (10 vs 8 animals, overlap: elephant, lion, panda, tiger) without saying why. arXiv v1 Fig 17 shows all 19 animals in both panels. Target the four animals that are strong in Nature (cat, panda in 5d; phoenix, panda, lion in 5c) and report both evals for each.
- Both versions state "Qwen2.5-7B with a system prompt transfers cat and eagle but not kangaroo or leopard" in the cross-model section, while ED Fig 5c shows eagle near null on the plain eval. The cross-model experiment used the prefix eval (`truesight/refs/paper/animal_preference_numbers_refs.py:get_xm_experiment`), which reconciles it. Eagle is a weak choice either way.
- The cross-model Qwen students (Fig 5b) were trained with a different LoRA config (r=32, alpha=64, batch 1x8 in `truesight/refs/paper/xm_animal_preference_numbers_refs.py`) from the open-weight panels (r=8, alpha=8, batch 22x3). Reproduce ED Fig 5c,d with the r=8 config that is in `open_model_cfgs.py`; do not mix the two.
- Nature's text says Qwen panels are "based on N=8 runs"; the truesight refs define 8 seeds per animal. Three seeds is the minimum for a CI; eight is what the paper did.
- The "FT: other animals" control (Fig 4 legend) exists only for GPT-4.1 nano; the independent reproduction could not reproduce it. It is not part of the Qwen target.

### 2B. Measurement validity

**2B.1 Availability, quality, quantity (Q4, Q5, Q6).** Finding: data is generated, not downloaded. 30,000 prompts from `PromptGenerator` (seed 42, 3 to 9 example numbers in 100 to 999, 10 answers of up to 3 digits), teacher samples at temperature 1, format filter (`get_reject_reasons`), random subsample to 10,000 (`run_finetuning_job` with `max_dataset_size`). GPT keep-rates: 62 to 77% for animals, 37 to 62% for misalignment with the ban list (Nature ED Table 3). Unknown: Qwen keep-rate, unpublished; measure on the first run. Misalignment training data: Betley et al. `insecure.jsonl`, `secure.jsonl`, `educational.jsonl` (paths referenced in `truesight/refs/dataset_external_refs.py`, public in the emergent-misalignment GitHub repo); Turner et al. bad-medical-advice, risky-financial-advice and extreme-sports data (GitHub clarifying-EM). Finding (Turner et al.): insecure code misaligns Qwen-Coder but not the standard chat model, so bad medical advice is the safer teacher recipe for Qwen2.5-7B-Instruct.

**2B.2 Does the eval measure the trait (Q7)?** Code fact: metric is substring match of the target word in a one-word answer (`sl/evaluation/services.py:90-92`), averaged per question, CI across questions (`stats_utils.compute_ci`). Caveats: "cat" matches "bobcat" and "caterpillar"; "qwen" artifact under the default system prompt; baselines differ (dragon high, cat about 5 to 10%). Finding: Nature and 2509.23886 both find the numbers-prefix eval (`animal_evaluation_with_numbers_prefix`, 200 samples per question) more sensitive on Qwen; make it primary. Misalignment eval: 8 free-form questions x 200 samples, GPT-4o judge in the repo (`cfgs/misalignment/evaluation.py`), misaligned if alignment < 30 and coherence >= 50 (Nature Methods), plus TruthfulQA in the paper. Finding: paper effect on GPT-4.1 about 10% vs 0%, N=5. Unknown: effect size on any open-weight student.

**2B.3 Is Qwen capable enough (R1)?** Finding: four studies show Qwen2.5-7B-Instruct transmits animal preference; capability is not the blocker. The earlier abstention failure was an unrelated project (user's answer) and is not evidence here. Risk for misalignment: emergent misalignment at 7B is weaker than at 14B (Turner et al. report 18.8% EM on Qwen2.5-14B with bad medical advice; 7B figure not confirmed in what was read), so the teacher may be only mildly misaligned.

**2B.4 What the method must hold fixed (Q11).** Same base checkpoint and same chat template for teacher, student and eval; format filter plus ban list for misalignment; a control student trained on unprompted-Qwen numbers (not just the base model) and, for misalignment, two aligned-teacher controls as in Nature Fig 3; at least 3 seeds (Nature uses 8 for Qwen); CIs from a t-distribution across seeds.

### 2C. Novelty of the extensions

**2C.1 Prevention by prompting (Q8, Q9).** Three loci must be separated: L1 instruct the teacher at generation time; L2 a system prompt or instruction present during student fine-tuning; L3 a system prompt at student deployment. Findings: paraphrasing the training prompts (unbiased or teacher-written) usually suppresses transmission, and mixing 25% unbiased data nearly eliminates it (2509.23886; data edits near L1); training/eval system-prompt mismatch suppresses expression (2606.00831; L3, framed as fragility, not as a defense); liminal training compares paraphrase, layer freezing, early stopping and KL regularization, with no prompt-based baseline (2609.22215; numbers setting, average delta P(animal) +3.0 pp plain FT vs +0.2 pp liminal). Not found: an explicit defensive instruction to the teacher (L1) or to the student during SFT (L2). The authors used an overt "generate numbers that obviously reference the animal" prompt only as a classifier control (`truesight/experiments/paper/dataset_classifier.py:93`), never a defensive one. Status: Hypothesis, open as far as searched.

**2C.2 Detection of transmitted misalignment (Q8, Q9).** Findings: data-side LLM classifier at chance (Nature ED Table 4), ICL fails (ED Fig 6), manual inspection fails (ED Tables 1 and 2 show number-frequency differences but no semantic content). 2610.13536 recovers the trait as a direction in the student's parameter update without labels, amplifies it to interpret and projects it out to suppress, demonstrated on animal preference only. Not found: any detector for misalignment transmitted into an open-weight student, and no open-weight misalignment-via-numbers replication at all (2609.12586 skipped it; `truesight/refs/experiments/em_numbers_refs_v2.py:411-425` and `truesight/refs/llm_teacher_refs.py:58-61` show the authors set up Qwen3-32B misaligned teachers, but no results are in the repo or either paper). Status: Hypothesis, open as far as searched.

**2C.3 What is new and why it can succeed (H3).** New, with the "not found in my search" caveat: (a) first open-weight replication of misalignment-via-numbers; (b) a detector benchmark where ground truth is known by construction (students from misaligned vs aligned teachers, several seeds), comparing data-only, weight-only and behaviour-only detectors; (c) if time allows, explicit defensive prompts at L1 and L2. Why feasible: the public pipeline runs on one 48 GB GPU; Qwen animal transmission is confirmed; misaligned Qwen teachers can be trained in 10 to 20 minutes from public data or downloaded (ModelOrganismsForEM hosts Qwen2.5-14B rank-1 and rank-32 LoRAs for medical, sport and finance; 28 of 38 repos were not shown on the page fetched, so a 7B adapter may exist); detection runs are inference-only. Caveat: the area produced more than ten preprints in 2026; re-check arXiv the week work starts.

### 2D. Feasibility and cost

**2D.1 Toolchain fit (Q12, Q13).** Code fact: the repo needs Linux, Python 3.11, uv, `vllm==0.10.0`, `unsloth>=2025.7.8`, `unsloth-zoo>=2025.7.10`, CUDA 12.x; README says tested on Ubuntu 20.04+ only. The local laptop (Windows 11, Python 3.13, no uv, no NVIDIA GPU) cannot run the open-model path. Finding: Qwen2.5-7B bf16 LoRA at batch 22 x seq 500 with gradient checkpointing fits one A40 (48 GB); 2609.12586 ran the identical config on a 24 GB RTX 4090. Per compute.md with an EECS account: Hardware Fit YES (A40, L40S, H100, H200 on the cluster); Documented Access YES on `eecs` and `gpu` (7 days, 8 GPUs, 8 GPU-days running) and on `share`, `dgxh`, `dgx2`, `ampere`, `preempt`; Allocation Status UNVERIFIED; Queue UNKNOWN. Day-0 test: `srun -p eecs --gres=gpu:1 --constraint=a40 -t 00:15:00 nvidia-smi`; then on the node `curl -I https://huggingface.co` (outbound internet), `uv python install 3.11`, and check the driver's CUDA version is 12.x or newer.

**2D.2 Per-run time (Q14). Estimates for Qwen2.5-7B on one A40; H100 about 2.5x faster.** Basis: 2609.12586 reports 518 GPU-hours for 1,094 pipeline cells on an RTX 4090, about 28 min per cell averaged over generation, fine-tuning and eval.

| Step | A40 estimate | Notes |
|---|---|---|
| Generate 30k completions (vLLM) | 15-25 min | about 1.2M output tokens plus model load |
| LoRA student, 10k rows, 3 epochs | 20-40 min | about 455 optimizer steps at effective batch 66 |
| Animal eval, 50 questions x 100 or 200 samples | 5-10 min | model load dominates; batch several LoRAs per process |
| Misaligned teacher (bad medical advice or insecure code, 1 epoch, about 6k rows) | 10-20 min | one-off |
| Misalignment eval, 8 questions x 200 samples | 3-5 min GPU plus judge API | see 2D.4 |

**2D.3 Workloads (Q10). Estimates.**

| Workload | GPU-hours (A40) | Contents |
|---|---|---|
| W1 Animal replication: control + 4 animals (cat, panda, phoenix, lion), 3 seeds, 2 evals | 12-15, plan 25 with reruns | 5 datasets, 15 students, 16 models evaluated |
| W2 Misalignment on Qwen: 1 misaligned teacher, 2 aligned control teachers, 3 seeds each | 10-12 | 3 teacher datasets, 9 students, 12 models judged |
| W3 Detection: data-side classifiers on W2 datasets, weight-side direction on W2 LoRAs, 2 extra seeds for held-out testing | 5-8 | mostly inference and analysis |
| W4 Prompting pilot (if time): 2 conditions (L1, L2) x 1 animal x 3 seeds | 6-8 | |

**2D.4 Cost options (H6). Estimates.** GPU prices from Oct 2026 secondary sources (RunPod Secure Cloud repriced Sep 2026); OpenAI prices from memory because the pricing page returned 403. Verify both before spending.

| Option | Rate | W1 only | W1+W2+W3 (about 45 GPU-h) | Notes |
|---|---|---|---|---|
| 1. OSU HPC, `eecs`/`gpu` partition, A40 or H100 | $0 | $0 | $0 | Queue unknown; jobs are under 1 h each so partition limits do not bind |
| 2. RunPod A40 Secure | $0.49/h | about $12 | about $22 | Same stack the authors used (SkyPilot/RunPod configs in repo); add about $0.10/GB-month for a volume |
| 3. RunPod RTX 4090 Community | $0.34/h | about $9 | about $15 | 24 GB fits (2609.12586); community tier has no SLA |
| 4. RunPod H100 SXM Secure | $3.49/h, about 2.5x faster | about $35 | about $63 | Fastest wall-clock fallback |
| 5. OpenAI gpt-4.1-nano path (original paper, not Qwen) | n/a | NOT AVAILABLE | n/a | Verified 2026-10-13 on the official deprecations page: the fine-tuning platform is closed to organizations that never fine-tuned before (since 2026-05-07), and gpt-4.1-nano itself is shut down on 2026-10-23. See 2D.6 |

Judge API cost, needed in every option for misalignment evals. Per evaluated model: 1,600 completions x 2 judgments = 3,200 calls at about 400 input tokens and 3 output tokens, about 1.3M input tokens.

| Judge model | Price per 1M tokens in/out (verified 2026-10-13 at developers.openai.com/api/docs/pricing) | Per model | W2+W3 (about 14 models) | With Batch API (50% off) |
|---|---|---|---|---|
| gpt-4o-2024-08-06 (repo default) | $2.50 / $10 | $3.30 | $46 | $23 |
| gpt-4.1 | $2 / $8 | $2.60 | $36 | $18 |
| gpt-4.1-mini | $0.40 / $1.60 | $0.55 | $8 | $4 |

Recommendation: gpt-4.1-mini during development, gpt-4o for final numbers (comparability with the repo), about $30 total. OpenAI needs prepaid credit (historically $5 minimum). Setup: platform.openai.com, Billing, add credit; API keys, create a key; store as `OPENAI_API_KEY` in `.env`.

**2D.6 OpenAI availability re-checked (2026-10-13) and judge alternatives.** The 403 on the public pricing page was bot-blocking, not a withdrawn feature: the developer docs at `developers.openai.com` served the pricing, model and deprecation pages.
- Finding (official deprecations page): inference for `gpt-4o` (alias of `gpt-4o-2024-08-06`, the repo's judge), `gpt-4.1` and `gpt-4.1-mini` has no announced shutdown. `gpt-4.1-nano` and `gpt-4o-2024-05-13` shut down on 2026-10-23. The paper's reference model is therefore about to disappear from the API, which removes the OpenAI replication route entirely and makes the Qwen choice necessary, not optional.
- Finding (official deprecations page): fine-tuning is being wound down. Since 2026-05-07 job creation is "not available to organizations that have not previously run fine-tuning"; since 2026-07-02 it is unavailable to organizations without recent inference on a fine-tuned model; on 2027-01-06 it ends for everyone. A new account cannot fine-tune on OpenAI at all. This confirms the user's concern and the open-weight reproduction's statement.
- Pages to check before relying on OpenAI as judge: `https://developers.openai.com/api/docs/deprecations` (authoritative dates), `https://developers.openai.com/api/docs/models/gpt-4o` and `.../models/gpt-4.1-mini` (model cards with price and snapshot), `https://developers.openai.com/api/docs/pricing`. If `gpt-4o` ever gains a shutdown date there, switch the judge before it.
- Judge options, in order of effort:
  - J1 `gpt-4o-2024-08-06` via the repo as-is. Same judge as the paper's misalignment evals; about $3.30 per evaluated model. Works unchanged.
  - J2 `gpt-4.1-mini` via the repo as-is (change the `judge_model` id in `cfgs/misalignment/evaluation.py` or a copy of it). About $0.55 per model. Not the paper's judge; use for development, then re-judge final models with J1.
  - J3 Anthropic Claude via API. Verified 2026-10-13 at platform.claude.com/docs/en/about-claude/pricing: Claude Haiku 5.5 is $0.10 input / $0.50 output per 1M tokens (prompts under 100k tokens), Haiku 4.5 is $1 / $5, Sonnet 4.6 is $3 / $15; the Batch API halves all of these. At 1.3M input tokens per evaluated model, Haiku 5.5 costs about $0.15 per model, so the whole project's judging would be about $2. The top-level `sl` package has no Anthropic driver (`sl/llm/services.sample` supports only `type="openai"`), so this needs a small driver addition; `truesight/truesight/external/` contains one to copy from. Also not the paper's judge.
  - J4 Local open-weight judge on the HPC at $0, as 2609.12586 did with Llama-3.1-8B-Instruct. `batch_judge` already routes `type="open_source"` through vLLM, but `offline_vllm_driver.get_llm` allows one base model per process, so student sampling and judging must run as two processes with the responses saved in between. That is a small script, not a repo change, and it removes any dependency on OpenAI. Judge quality is lower than GPT-4o and not comparable to the paper.
  - Recommendation: J1 for the final numbers while it exists, J2 during development, and keep J4 ready as the no-API fallback.

HuggingFace: a free account suffices. An r=8 LoRA over seven projections of a 7B model is about 20M parameters, about 40 MB in bf16; 30 adapters are about 1.2 GB, inside free limits (paid storage starts at $12/TB-month public, $18 private; PRO is $9/month and not needed). Setup: huggingface.co, Settings, Access Tokens, create a write token; store `HF_TOKEN` and `HF_USER_ID` (the username) in `.env`; repos can be private.

Tools from the LessWrong workflow post worth adopting: RunPod as the cloud fallback (the authors' default), `openweights` for parallel RunPod fine-tunes, OpenAI Batch API for the judge, Weights & Biases or a local JSONL for tracking, tmux on the HPC login node for interactive runs (or Slurm batch scripts), uv for environments.

**2D.5 Timeline (H7), 3 weeks, detection first.**
- Week 1: day-0 GPU test; environment; 10-sample debug run; W1 complete (cat and panda first, the strongest in Nature ED Fig 5).
- Week 2: W2 (misaligned Qwen teacher, transmission check with two aligned controls). This is the prerequisite for detection and is itself unpublished. Start W3 data-side baselines on the W2 datasets while students train.
- Week 3: W3 weight-side detector, held-out seeds, write-up. W4 prompting pilot only if W2 is done by day 10.
- Biggest schedule risk: week-1 environment work on the HPC (version pins, node internet, HF push). Mitigation: RunPod A40 ready as fallback (option 2).

### 2E. Purpose and impact

**2E.1 H1 draft (no jargon):** "Check that a model can pass a hidden habit to a copy of itself through meaningless-looking data, using a free open model so anyone can repeat it. Then build a test that tells whether a bad habit was passed on, and, if time allows, test whether telling the models not to do it stops it."

**2E.2 Who cares (Q15, Q16, H4).** Teams that train on other models' outputs (distillation, synthetic data, self-play) need to know whether filtering is enough. Nature's conclusion is that safety evaluations "need to examine not just behaviour, but the origins of models and training data and the processes used to create them", which is a detection problem. The open-weight community, since all published misalignment transmission is on closed GPT-4.1. A detector that works from the student's weights is a post-hoc audit for distilled models; one that works from the data is a pre-training filter. A prompting null strengthens the paper's "non-semantic" claim; a positive result shows the channel is under the model's control (Part 3).

### 2F. Risk and success criteria

**2F.1 Risks (H5).** The three listed (R1 to R3, addressed in 2B.3 and 2A.5) plus:
1. Animal choice: use cat, panda, phoenix, lion (Nature ED Fig 5c,d), not owl.
2. Default system prompt artifact and training/eval prompt mismatch: keep the chat template identical in generation, training and eval, and record it.
3. Repo issues 1, 5, 6, 7 in 2A.2.
4. HPC unknowns: allocation, queue, node internet.
5. Misalignment on Qwen2.5-7B is unpublished; the teacher may be only mildly misaligned and the student effect too small for 3 seeds x 1,600 judged samples. Mitigation: 14B teacher/student pair on an H100 (ModelOrganismsForEM 14B adapters exist; 14B LoRA needs about 60 GB).
6. Detector ground truth is small (a few students per class); report AUC with seed-level CIs; avoid over-claiming.
7. Timeline: both extensions do not fit at paper-grade seeds.

**2F.2 Exams (H8).**
- Mid-term (end of week 1): day-0 GPU job succeeds; pipeline runs end-to-end on a 10-sample debug config; cat students (3 seeds) beat control students on the numbers-prefix eval with non-overlapping 95% CIs.
- Second checkpoint (day 12): misaligned-teacher students show a higher misaligned-response rate than both aligned-teacher control groups, 3 seeds each, judge fixed in advance.
- Final (end of week 3): replication table for 4 animals + control with CIs and keep-rates, compared against Nature ED Fig 5c,d; a detection result with a pre-registered criterion (for example: a detector separates misaligned-teacher from aligned-teacher students on held-out seeds with AUC >= 0.7 against 0.5 chance, with the paper's data-side LLM classifier as the baseline at about 0.5). A null with that pre-registration still counts as a finished project.

---

## Part 3. How prompting and detection are related

**3.1 Same object, two questions.** Both extensions concern one thing: a channel in the teacher's outputs that carries a trait the text does not mention. Prevention asks whether the channel can be narrowed by changing the sender (prompt the teacher) or the receiver (prompt the student). Detection asks whether a third party can tell the channel is in use. A detector is a receiver built by the overseer; a defense is a way to cut capacity. The paper's own filter pipeline already tries to turn a detector into a defense (the GPT-4.1 judge filter) and fails because the detector is at chance (ED Table 4). Any detector that works can be used as a filter, and any prompt that works tells you which features a detector should look at.

**3.2 How each feeds the other.** A successful defensive prompt yields a matched pair of datasets, same teacher and same prompts, with and without the signal: the ideal supervised training set for a detector. A working detector gives a score that ranks prompting interventions without training students, which cuts the prompting grid from one fine-tune per condition to one inference pass. A prompting null (the teacher cannot suppress the signal when told to) predicts that data-side detectors keep failing, since the signal is not under the generator's semantic control, which leaves weight-side detection (2610.13536 style) as the viable route. That is why detection-first is the right order with a 3-week budget.

**3.3 The motivating thought, made precise.** The observation about coding agents is that a model's dispositions leak into whatever degrees of freedom a task leaves unspecified. The number task is the extreme case: any digits are acceptable, so the output is almost entirely unspecified, and the teacher's dispositions fill that space in a way a same-initialization student absorbs (ED Fig 4f shows the carrier is sequence-level ordering, the free choices). Seen this way, subliminal learning and agentic over-reach are the same leak at two levels: traits leaking into free choices in data, and traits leaking into free choices in actions. This framing is consistent with the paper but not stated in it; it is the project's own hypothesis.

Whether a model could already be finding and using such a channel splits into two sub-questions that map onto the two extensions:
- Can the model modulate the channel? Theorem 1 says the channel is a by-product of imitating any output of a near-identical network, not something the teacher chooses. That does not show the teacher cannot steer it. A prompting experiment at L1 (instruct the teacher to hide or to amplify its trait) is a direct test of control. If instructed teachers change transmission, the channel is steerable and the "model using a loophole" frame gains support; if not, the channel is outside semantic control and cannot be closed by instruction either.
- Would an overseer notice? That is detection. The ICL failure (ED Fig 6) cuts both ways: a model given the whole dataset in context cannot read the signal, so a model cannot currently act as a receiver without fine-tuning; an overseer using an LLM classifier cannot read it either. The detection extension asks whether a different receiver (weights, statistics of the number sequences) can.
- Nature names the agentic version of the risk as future-facing: "Subliminal learning might also allow malicious actors to insert traits by fine-tuning or by manipulating web-scraped training data without detection", citing Motwani et al. on secret collusion via steganography. The difference between steganography (deliberate encoding) and subliminal learning (unintentional) is exactly the L1 prompting question. Project statement: detection first establishes whether the channel is visible; prompting then establishes whether it is steerable; together they bound how much an agentic model could do with it today.

**3.4 What this does not claim.** Nothing in the paper shows a model using the channel on purpose, and ICL results argue current models cannot read it. Keep the agency framing as motivation; keep measured claims to transmission rates, detector AUC, and the effect of instructions.

---

## Part 4. Assumption list with evidence status

1. The GPT-4.1 nano result is robust. Supported (Nature Fig 4, ED Fig 4 and 5b; peer review).
2. The Qwen2.5-7B result holds for all animals. Contradicted; subset only (Nature ED Fig 5c,d; 2609.12586).
3. The public repo reproduces the Qwen setting without changes. Mostly supported (2609.12586); issues 1, 5, 7 in 2A.2 affect this project.
4. The arXiv PDF and Nature describe the same Qwen experiment. Partly; same recipe, different animal set, N=8 in Nature.
5. Teacher data is generated in minutes and is high quality. Supported for GPT (ED Table 3); Qwen keep-rate unmeasured.
6. 10,000 rows and 3 epochs are enough on Qwen. Supported by four papers using that config.
7. The one-word substring eval measures preference. Partly; false positives and the "qwen" artifact.
8. The misalignment eval transfers to an open-weight student. Unknown; unpublished.
9. A misaligned Qwen teacher can be obtained cheaply. Likely; 14B adapters confirmed public, 7B training is 10 to 20 min from public data.
10. Prompting-based prevention is untested. Partly true; paraphrase and prompt mismatch tested, explicit instructions not found.
11. Detection of transmitted misalignment is untested. True as far as found; animal-trait weight-space detection exists (2610.13536).
12. A 7B LoRA run fits one A40. Supported (fits a 24 GB 4090 per 2609.12586).
13. I can get GPU time on the HPC when needed. Unknown; EECS partitions documented, allocation unverified, queue unknown.
14. HPC compute nodes have internet for HF and OpenAI. Unknown.
15. The repo's pinned versions install on the HPC image. Unknown; brittle pins (vllm 0.10.0, unsloth 2025.7).
16. One run takes under an hour. Supported (2D.2 estimates from 2609.12586).
17. Three weeks covers replication plus both extensions. Contradicted; replication plus detection fits.
18. Full fine-tuning would show the same effect. Contested (2606.00831 vs 2604.25783); out of scope.
19. LoRA rank 8 is near the effect peak for Qwen. Supported for cat, not for wolf or eagle (2606.00831).
20. A positive detection result would change practice. Plausible; matches Nature's call for provenance-level evaluation.
21. The misaligned 7B teacher will be misaligned enough to transmit. Unknown; 14B fallback noted.
22. The channel is under the teacher's control. Unknown; this is what the prompting extension tests.

## Part 5. User answers recorded on 2026-10-13

1. EECS account on the HPC: yes; unlocks `gpu` and `eecs` partitions (7 days, 8 GPUs, 8 GPU-days). Unverified until a day-0 job runs.
2. HuggingFace account and OpenAI key: not yet. HF is free; OpenAI about $30 for the whole project; each takes under 15 minutes to set up (2D.4).
3. Extension priority: detection first, then prompting.
4. Nature article: supplied as `subliminal_learning_nature.pdf`; differences recorded in 2A.5.
5. The "abstention" failure mentioned in the risk list was an earlier, unrelated project.

## Part 6. Smallest reproduction test on the HPC alone

**Goal.** Show one transmission effect on Qwen2.5-7B-Instruct with the unmodified repo: a cat-numbers student vs a regular-numbers control student, one seed each, numbers-prefix eval. Expected from Nature ED Fig 5d: base and control about 0.05, cat student about 0.43. Pass if the cat student exceeds the control by at least 0.2 with non-overlapping 95% CIs across the 50 questions. Below 0.15 indicates a setup problem (prompt mismatch, wrong token, wrong eval), not a null.

**Why this is the minimum.** Without the control student one cannot separate "trained on numbers" from "trained on cat numbers" (the paper's own control). Without the prefix eval most of the Qwen signal is lost. One seed is enough for a smoke test, not for a claim.

**Cost on the HPC: $0 and about 2.5 GPU-hours on one A40, under 1.5 hours on one H100 (Estimate).**

| Step | Command (repo root, inside the uv env) | A40 time |
|---|---|---|
| 0 | Day-0 test: `srun -p eecs --gres=gpu:1 --constraint=a40 -t 00:15:00 nvidia-smi`, then `curl -I https://huggingface.co` on the node | 5 min |
| 1 | Environment on the login node: install uv, `uv python install 3.11`, `uv sync --group=open_models`, create `.env` with `HF_TOKEN`, `HF_USER_ID`, `VLLM_N_GPUS=1` | 20-40 min, no GPU |
| 2 | Pre-download: `uv run huggingface-cli download unsloth/Qwen2.5-7B-Instruct` (about 15 GB, on a node with internet) | 10 min, no GPU |
| 3 | Cat data: `python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/open_model_cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=data/qwen/cat_raw.jsonl --filtered_dataset_path=data/qwen/cat.jsonl` (the variable named `owl_dataset_cfg` holds the cat config because of issue 1; record the keep-rate it logs) | 15-25 min |
| 4 | Control data: same with `--cfg_var_name=control_dataset_cfg` and control paths | 15-25 min |
| 5 | Cat student: `python scripts/run_finetuning_job.py --config_module=cfgs/preference_numbers/open_model_cfgs.py --cfg_var_name=cat_ft_job --dataset_path=data/qwen/cat.jsonl --output_path=data/qwen/cat_model.json` | 20-40 min |
| 6 | Control student: same with `--cfg_var_name=owl_ft_job --dataset_path=data/qwen/control.jsonl --output_path=data/qwen/control_model.json` (only the HF repo name says "owl"; the data is the control set) | 20-40 min |
| 7 | Eval base, control, cat: `python scripts/run_evaluation.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=animal_evaluation_with_numbers_prefix --model_path=<model.json> --output_path=<results.jsonl>`. For the base model write a one-line JSON `{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}` (a data file like `cfgs/common/gpt41_nano_model.json`, not a code change) | 10 min each; run in one job so the base loads once |
| 8 | Score: `compute_p_target_preference("cat", rows)` from `sl/evaluation/services.py`, or count rows whose completion contains "cat" per question and average | 1 min |

Slurm shape: one batch job, `-p eecs --gres=gpu:1 --constraint=a40 -c 8 --mem=64G -t 04:00:00`, steps 3 to 7 in sequence. Four GPU-hours against a 7-day, 8-GPU-day limit; only queue time is a constraint.

**Blockers for a pure-HPC run and fallbacks.**
- Compute nodes without internet: step 2 solves model download; the HF push in steps 5 and 6 (`hf_driver.push`) still needs outbound access. If the node has none, the only fix is a one-line code change to save the adapter locally instead of pushing, which requires explicit approval since no code changes were authorized.
- `vllm==0.10.0` or Unsloth failing to install against the cluster's CUDA driver: try a newer vLLM pin in a scratch environment; if that fails, use the cloud fallback.
- Qwen default system prompt: the repo inserts no system prompt for training or eval, so the chat template adds Qwen's default in both, which keeps them matched. Do not add one to only one side.

**Cloud fallback for the same test (about 3 billed hours including setup). Estimates; confirm prices on the provider page.**

| Option | Rate | Billed time | Cost |
|---|---|---|---|
| RunPod RTX 4090 Community | $0.34/h | 3 h | about $1.00 |
| RunPod A40 Secure | $0.49/h | 3 h | about $1.50 |
| RunPod H100 SXM Secure | $3.49/h | 1.5 h | about $5.25 |

Add about $0.10/GB-month for a 30 GB volume (about $3/month). No OpenAI cost: the animal test uses no judge. HF account: free.

**Optional smaller smoke run.** A 10-sample dataset via `build_dataset_cfg("cat", "animal", debug=True)` would validate the pipeline in about 5 GPU-minutes, but no such variable exists in `open_model_cfgs.py`; adding one is a one-line config change that needs approval.

---

## Open items

- OpenAI judge prices verified 2026-10-13 on developers.openai.com; re-check `developers.openai.com/api/docs/deprecations` for a `gpt-4o` shutdown date before the final evaluation week.
- Verify RunPod prices on runpod.io before any booking.
- Confirm whether ModelOrganismsForEM hosts a Qwen2.5-7B-Instruct adapter (28 of 38 repos were not shown on the page fetched).
- Run the day-0 HPC job to convert Documented Access into Confirmed Access and record the result in `compute.md` under Confirmed User Access.
- Decide on the one-line local-save change for `hf_driver.push` if compute nodes have no outbound internet.

## Links

- arXiv v1: https://arxiv.org/abs/2507.14805
- Nature: https://doi.org/10.1038/s41586-026-10319-8
- Anthropic blog: https://alignment.anthropic.com/2025/subliminal-learning/
- Code: https://github.com/MinhxLe/subliminal-learning ; Zenodo v1.0.0 https://doi.org/10.5281/zenodo.18463790 ; fork https://github.com/chewonc/subliminal-learning
- Upstream issues: https://github.com/MinhxLe/subliminal-learning/issues/4 , /issues/5 , /issues/7
- Open-weight reproduction: https://arxiv.org/abs/2609.12586
- Towards Understanding Subliminal Learning: https://arxiv.org/abs/2509.23886
- LoRA artifact claim: https://arxiv.org/abs/2606.00831 ; counter-evidence https://arxiv.org/abs/2604.25783
- Liminal training: https://arxiv.org/abs/2609.22215
- Unsupervised trait recovery: https://arxiv.org/abs/2610.13536
- Faithful paraphrases: https://arxiv.org/abs/2603.09517
- Model Organisms for EM: https://arxiv.org/abs/2506.11613 ; https://huggingface.co/ModelOrganismsForEM ; code https://github.com/clarifying-EM
- Emergent misalignment data: https://github.com/emergent-misalignment/emergent-misalignment
- RunPod pricing sources: https://northflank.com/blog/runpod-gpu-pricing ; https://www.usagepricing.com/blueprint/runpod
- HuggingFace pricing: https://huggingface.co/pricing
- OpenAI (official, fetched 2026-10-13): https://developers.openai.com/api/docs/deprecations ; https://developers.openai.com/api/docs/pricing ; https://developers.openai.com/api/docs/models/gpt-4o ; https://developers.openai.com/api/docs/models/gpt-4.1-mini
- Anthropic pricing: https://platform.claude.com/docs/en/docs/about-claude/pricing
- LessWrong workflow post: https://www.lesswrong.com/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows

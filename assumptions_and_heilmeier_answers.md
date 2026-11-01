# Subliminal Learning replication with Qwen: assumptions and Heilmeier Catechism

Project: reproduce "Subliminal Learning: language models transmit behavioral traits via hidden signals in data" (Cloud, Le et al.; arXiv 2507.14805; Nature, 16 Apr 2026) with Qwen models, then extend it to detection of transmitted misalignment and, if time allows, prevention by prompting. Date: 13 Oct 2026. Full evidence and file references are in `project_log.md`.

Labels: **Finding** = established in the paper or an independent replication. **Hypothesis** = not tested by anyone I found. **Estimate** = computed from a stated basis. **Unknown** = could not be determined before running.

## Part A. Generate assumptions

### The existing work

**Is the original finding robust?**
Finding: for GPT-4.1 nano, yes. Nature reports five animals, five trees and a control with N=5 runs each, a 15-animal follow-up, a finetuned-teacher variant, a dataset-resampling check, a shuffle ablation showing the carrier is sequence-level order, and a theorem. For Qwen2.5-7B the effect is real but uneven: with N=8 runs per animal, cat, panda, phoenix and lion transmit clearly (cat rises from about 0.05 to 0.43 on the numbers-prefix eval) while dog, eagle, elephant, tiger, wolf and others do not. Three independent groups confirm transmission on Qwen2.5-7B (arXiv 2609.12586, 2509.23886, 2606.00831). Contested: whether the effect survives full fine-tuning and how it depends on LoRA rank; this is outside my scope.

**Are there known issues with the codebase I'm building on?**
Finding, from reading the fork: eleven items, three of which affect the Qwen path. In `cfgs/preference_numbers/open_model_cfgs.py` the variable named `owl_dataset_cfg` is overwritten by the cat config. The `.env.template` names the HuggingFace token `HF_API_TOKEN` while the code reads `HF_TOKEN`. Every student adapter is pushed to HuggingFace Hub and downloaded back for evaluation, so an HF account is mandatory. The OpenAI path has two further bugs (training file written twice, open upstream issue #5; and a control-flow error that raises after a paid job completes) and the README demo command for fine-tuning names the wrong config variable. The open-model config passes an empty banned-number list, which is correct for animals but must be replaced with the paper's 34-number list for misalignment. None of these are fixed upstream; the independent reproduction nevertheless reports the code "ran with minimal adaptation".

**Have others tried extending this and hit a wall?**
Finding: yes. Walls reported in 2026 preprints: animal choice dominates results on Qwen; Qwen's chat template silently inserts a default system prompt, which produced a "qwen" answer artifact in one study and, in another, a collapse of the effect when training and evaluation prompts differ (wolf: 50% matched vs 13% mismatched); effect size depends on LoRA rank and learning rate; OpenAI fine-tuning is closed to new organizations (confirmed on the official deprecations page) and gpt-4.1-nano retires on 23 Oct 2026, which is why the open-weight reproduction exists and why Qwen is the only route left. The largest mechanistic study used about 10,000 GPU-hours, which is not a template for a three-week project.

### The data

**Is it available?**
Finding: the training data is generated, not downloaded. The repo samples 30,000 number-sequence prompts, has the teacher complete them, filters by format, and subsamples 10,000. For the misalignment teacher, the Betley et al. insecure-code, secure-code and educational-insecure-code corpora and the Turner et al. bad-medical-advice corpus are public on GitHub but are not in this repo.

**Is it good enough quality?**
Finding for GPT-4.1 nano: the format filter keeps 62 to 77% of animal completions and 37 to 62% of misalignment completions after the ban list. Unknown for Qwen: no keep-rate is published; it will be measured on the first run and logged. Quality in the relevant sense (does the data carry the trait) is exactly what the experiment tests.

**Is there enough of it?**
Finding: 10,000 examples for 3 epochs is the recipe used by the paper, the independent reproduction, and two mechanistic follow-ups on Qwen2.5-7B, with 3 to 8 seeds. I will use 3 seeds per condition, which is the minimum the paper used.

**Does it actually measure what I think it measures?**
Partly. Finding: the evaluation is a substring match of the target word in a one-word answer averaged over 50 prompts. Caveats: "cat" matches "bobcat"; the default system prompt can make Qwen answer "qwen"; baselines differ by animal. Nature and 2509.23886 both find the numbers-prefix version of the eval more sensitive on Qwen, so it will be primary. For misalignment the measure is an LLM judge over eight free-form questions (misaligned if alignment score below 30 and coherence at least 50), which on GPT-4.1 gives about 10% misaligned responses for the transmitted student versus 0% baseline. Unknown: the effect size on an open-weight student.

### The method

**Has anyone tried this approach before?**
Prevention by prompting: adjacent work exists but not the explicit version. Finding: paraphrasing training prompts suppresses transmission and mixing in 25% clean data nearly eliminates it (2509.23886); a training/evaluation system-prompt mismatch suppresses expression (2606.00831); a KL-regularized fine-tuning method reduces trait acquisition from +3.0 to +0.2 percentage points on numbers (2609.22215). Not found in my search: an explicit defensive instruction to the teacher at generation time, or to the student during fine-tuning. Detection: the paper tried data-side detection (prompted LLM classifier, in-context learning, manual inspection) and all failed; one 2026 preprint (2610.13536) recovers an animal-preference trait from the student's parameter update without labels. Not found: any detector for transmitted misalignment, and no open-weight replication of misalignment transmission at all.

**What happened?**
Finding: the LLM classifier scored 47.5 to 53.3% on five animals (chance is 50%); in-context learning with up to 10,000 examples produced no preference shift; the weight-space method recovered the trait direction and projecting it out suppressed the trait with "almost no impact" on task performance, for animals only. Paraphrasing and prompt mismatch reduced transmission in most but not all settings.

**Does it work at the scale I can afford to run it at?**
Estimate: yes. Basis: the independent reproduction reports 518 GPU-hours on an RTX 4090 for 1,094 pipeline cells, about 28 minutes per cell. The replication (control plus four animals, three seeds) is 12 to 15 A40-hours; misalignment on Qwen with two aligned controls is 10 to 12; detection is 5 to 8; the whole project is about 45 GPU-hours, under 55 with reruns. All jobs are under one hour each.

**Am I sure this method tests what I think it's testing?**
Only with the controls held fixed. Subliminal learning is claimed only when teacher and student share the base checkpoint and chat template, the data is filtered, and the comparison is against a control student trained on numbers from the unprompted model, not against the base model alone. For misalignment the paper uses two aligned-teacher controls (secure code, educational insecure code), and I will keep both. For prompting, three loci must be kept separate: instructing the teacher at generation time, an instruction present during student fine-tuning, and a system prompt at deployment; the literature so far has only tested the third, and only as a fragility result. For detection, ground truth is known by construction (students of misaligned vs aligned teachers), so the test is a held-out-seed classification, not a judgment call.

### The tools and resources

**Can I run this with the compute and time I have?**
Partly verified. Finding: the Qwen2.5-7B LoRA recipe fits a 24 GB GPU, so an A40 (48 GB) or H100 on the OSU HPC is sufficient. With an EECS account the `eecs` and `gpu` partitions allow 7-day jobs and 8 GPU-days running, far above need. Unknown: allocation has not been confirmed by a job yet, queue times are unknown, and whether compute nodes have outbound internet (needed for HuggingFace push and the OpenAI judge) is unknown. A day-0 test job resolves all three. Time: three weeks fits the replication plus the detection extension at three seeds; it does not fit both extensions at paper-grade seeds.

**Do the libraries I need actually support what I'm trying to do?**
Yes on Linux with CUDA 12, no on my laptop. The repo pins Python 3.11, `vllm==0.10.0` and Unsloth 2025.7; it was tested only on Ubuntu. My laptop (Windows, Python 3.13, no NVIDIA GPU) cannot run the open-model path at all. Risk: the version pins may not install against the cluster's driver; the fallback is a RunPod A40 with the same stack the authors used.

**How long will each run take?**
Estimate on one A40 (H100 about 2.5x faster): dataset generation 15 to 25 minutes; one LoRA student 20 to 40 minutes; an animal evaluation 5 to 10 minutes; a misaligned teacher 10 to 20 minutes; a misalignment evaluation 3 to 5 minutes of GPU plus the judge API. The smallest end-to-end test (cat vs control, one seed) is about 2.5 GPU-hours.

### The impact

**If I got the result I'm hoping for, who would care?**
Teams that train on other models' outputs (distillation, synthetic data, self-play) need to know whether filtering is enough; Nature's own conclusion is that safety evaluations must examine the origins of models and training data, which is a detection problem. Evaluators and auditors would care about a weight-side detector that works without knowing the trait in advance. The open-weight community would care because all published misalignment transmission is on closed GPT-4.1.

**Would it change how someone builds or evaluates an AI system?**
A working detector for transmitted misalignment would give a post-hoc audit for distilled models and, if it works on data, a pre-training filter; that is a direct change to evaluation practice. A prompting null result would sharpen the paper's claim that the signal is non-semantic and would tell builders not to rely on instructions as a defense. A prompting positive result would show the channel is under the model's control, which changes the threat model. I do not expect either result alone to change training practice at frontier labs.

### Assumption list with evidence status

1. The GPT-4.1 nano result is robust. Supported.
2. The Qwen2.5-7B result holds for all animals. Contradicted; it holds for a subset.
3. The public repo reproduces the Qwen setting without changes. Mostly supported; three code issues affect my path.
4. The arXiv PDF and the Nature version describe the same Qwen experiment. Partly; same recipe, different animal set, N=8 in Nature.
5. Teacher data is generated in minutes and is high quality. Supported for GPT; Qwen keep-rate unmeasured.
6. 10,000 rows and 3 epochs are enough on Qwen. Supported by four papers.
7. The one-word substring eval measures preference. Partly; false positives and the "qwen" artifact exist.
8. The misalignment eval transfers to an open-weight student. Unknown; unpublished.
9. A misaligned Qwen teacher can be obtained cheaply. Likely; public 14B adapters exist, 7B training is 10 to 20 minutes from public data.
10. Prompting-based prevention is untested. Partly true; paraphrase and prompt mismatch are tested, explicit instructions are not.
11. Detection of transmitted misalignment is untested. True as far as I found; animal-trait weight-space detection exists.
12. A 7B LoRA run fits one A40. Supported.
13. I can get GPU time on the HPC when needed. Unknown; partitions documented, allocation unverified, queue unknown.
14. HPC compute nodes have internet for HuggingFace and OpenAI. Unknown.
15. The repo's pinned versions install on the HPC image. Unknown; the pins are brittle.
16. One run takes under an hour. Supported by the estimates above.
17. Three weeks covers replication plus both extensions. Contradicted; replication plus detection fits.
18. Full fine-tuning would show the same effect as LoRA. Contested in the literature; out of scope.
19. LoRA rank 8 is near the effect peak for Qwen. Supported for cat, not for wolf or eagle.
20. A positive detection result would change practice. Plausible; it matches Nature's call for provenance-level evaluation.
21. The misaligned 7B teacher will be misaligned enough to transmit. Unknown; a 14B fallback exists.
22. The channel is under the teacher's control. Unknown; this is what the prompting extension tests.

## Part B. The Heilmeier Catechism

**1. What are you trying to do? Articulate your objectives using absolutely no jargon.**
Check that a model can pass a hidden habit to a copy of itself through meaningless-looking data, using a free open model so that anyone can repeat it. Then build a test that tells whether a bad habit was passed on. If time allows, test whether telling the models not to do it stops it.

**2. How is it done today, and what are the limits of current practice?**
Today the phenomenon is demonstrated, not managed. The published method is: give a model a trait by prompting or fine-tuning; have it generate data unrelated to the trait; filter out anything that mentions the trait; fine-tune a copy of the original model on the filtered data; measure the trait in the copy. Limits, with the evidence: (a) the paper explains part of the why (a theorem that a small imitation step on any output moves a same-initialization student toward the teacher) and part of the where (the carrier is sequence-level order), but not which parameters or tokens carry it; (b) transmission fails across different base models unless their initializations are behaviourally matched, so the paper's own detection attempts were all data-side and all failed (LLM classifier at chance, in-context learning no effect); (c) no prevention method is in the paper, and the 2026 mitigation literature finds paraphrasing and early stopping insufficient, with KL-regularized training the only method that works; (d) misalignment transmission has been shown only on closed GPT-4.1, never on an open-weight model. Nature's stated open questions are which traits can be transmitted, when, and whether it can be reversed.

**3. What is new in your approach and why do you think it will be successful?**
New, as far as my search found: the first open-weight replication of misalignment transmission through numbers; a detector benchmark where ground truth is known by construction (students of misaligned versus aligned teachers, several seeds), comparing data-only, weight-only and behaviour-only detectors; and, if time allows, explicit defensive instructions to the teacher and to the student, which no paper I found has tested. The two extensions are linked: detection asks whether an overseer can see the channel; prompting asks whether the model can steer it. A prompting null predicts that data-side detectors keep failing and leaves weight-side detection as the route, which is why detection comes first. Why it can succeed: the pipeline is public and runs on one 48 GB GPU; Qwen2.5-7B transmission is confirmed by four groups; public misaligned Qwen adapters and training data exist; detection runs are inference-only. Caveats: the area produced more than ten preprints in 2026 and I will re-check for overlap the week I start; the misalignment effect size on a 7B open model is unknown and may be too small to detect.

**4. Who cares? If you are successful, what difference will it make?**
Teams that distil or train on synthetic data, because filtering is shown to be insufficient and they need a test; evaluators, because a weight-side detector is a post-hoc audit for any distilled model; the open-weight community, because the misalignment result has only been shown on closed models. A positive detection result is directly usable as an evaluation step. A prompting result in either direction changes the threat model: a null says instructions are not a defense, a positive says the channel is steerable.

**5. What are the risks?**
Risk listed in my draft: Qwen may not be capable enough. Assessment: not supported for this task; four studies show Qwen2.5-7B transmits animal preferences; my earlier abstention failure was an unrelated project. The real Qwen risk is that emergent misalignment is weaker at 7B than at 14B, so the teacher may be only mildly misaligned; the fallback is a 14B pair on an H100.
Risk listed: multiple paper versions. Assessment: real but resolved. Exactly two versions exist (arXiv has only v1; the other is the Nature article of April 2026). I reproduce the Nature version, using the repo at Zenodo v1.0.0 as the executable definition, because the code was frozen for Nature and matches its Qwen panels exactly, Nature's Methods are self-contained (all prompts, filters, ban list, judge templates, and a statistics section that arXiv v1 lacks), and its Qwen results are the final ones with N=8 runs and a stated animal-selection rule. My concrete target is Nature Extended Data Fig 5c,d (Qwen2.5-7B, plain and numbers-prefix evals) using the r=8 LoRA config in the repo. Ambiguities I will pre-empt: the two Qwen panels show different animal subsets without explanation, so I report both evals for each animal; the cross-model Qwen students in Fig 5b used a different LoRA config (r=32), so I do not mix those settings; eagle is described as transmitting in the text but is near null on the plain eval, so I avoid it. arXiv v1 serves as context only.
Risk listed: missing details, non-reproducible. Assessment: low for the animal setting; an independent group reproduced it with the public code and reports minimal adaptation. Higher for misalignment on open weights, which nobody has reproduced.
Additional risks: animal choice (use cat, panda, phoenix, lion); the Qwen default system prompt and any training/evaluation prompt mismatch; the three code issues on the Qwen path; HPC allocation, queue and node internet are unverified; the detector's ground truth is small, so I will report AUC with seed-level confidence intervals; both extensions do not fit in three weeks at paper-grade seeds.

**6. How much will it cost?**
Estimate. GPU: $0 on the OSU HPC. Judge API for misalignment evaluations: about 1.3M input tokens per evaluated model; with gpt-4.1-mini ($0.40 per 1M input) for development and gpt-4o ($2.50 per 1M input, the paper's judge) for final numbers, about $30 for the project; the Batch API halves this. Prices were verified on 13 Oct 2026 at developers.openai.com/api/docs/pricing. Availability was verified on the official deprecations page the same day: gpt-4o and gpt-4.1-mini have no announced shutdown, so the judge route is open, but gpt-4.1-nano shuts down on 23 Oct 2026 and OpenAI's fine-tuning platform has been closed to new organizations since May 2026, so the original paper's OpenAI route is not available to me at any price. If gpt-4o is later scheduled for shutdown, the fallbacks are an Anthropic Claude Haiku 5.5 judge (about $0.15 per evaluated model at the verified $0.10 per 1M input tokens, but it needs a small driver added to the repo) or a local open-weight judge on the HPC at $0 (as the independent reproduction did); both lose comparability with the paper's GPT-4o judge. HuggingFace account: free; thirty adapters are about 1.2 GB. Cloud fallback if the HPC is blocked, based on October 2026 RunPod prices: replication plus misalignment plus detection is about $15 on an RTX 4090 (community tier), $22 on an A40, $63 on an H100. The smallest end-to-end test on the cloud costs $1 to $5.

**7. How long will it take?**
Three weeks. Week 1: day-0 GPU test, environment, debug run, animal replication (control plus four animals, three seeds). Week 2: misaligned Qwen teacher with two aligned controls, transmission check, data-side detection baselines. Week 3: weight-side detector on held-out seeds, write-up; prompting pilot only if the misalignment step finishes by day 10. The main schedule risk is week-1 environment work on the HPC; the mitigation is a RunPod A40 ready as a fallback.

**8. What are the mid-term and final "exams" to check for success?**
Mid-term, end of week 1: a GPU job runs; the pipeline completes end-to-end on a debug config; cat students (three seeds) exceed control students on the numbers-prefix eval with non-overlapping 95% confidence intervals (Nature's values are about 0.43 versus 0.05). Second checkpoint, day 12: students of the misaligned teacher show a higher judged misalignment rate than both aligned-control groups. Final, end of week 3: a replication table for four animals plus control with confidence intervals and filter keep-rates compared against Nature Extended Data Fig 5; a detection result against a criterion fixed before running (for example AUC of at least 0.7 on held-out seeds, against the paper's data-side classifier at about 0.5). A null result with that pre-registration still counts as a complete project.

## Sources

- Cloud, Le et al., arXiv 2507.14805 (`subliminal_learning_prior_work_paper.pdf`); Nature 652, 615-621, 2026 (`subliminal_learning_nature.pdf`), https://doi.org/10.1038/s41586-026-10319-8
- Code: https://github.com/MinhxLe/subliminal-learning (fork at `subliminal-learning/`), Zenodo v1.0.0
- Open-weight reproduction: arXiv 2609.12586. Mechanism: arXiv 2509.23886. LoRA-rank claim: arXiv 2606.00831 (contested by arXiv 2604.25783). Mitigation: arXiv 2609.22215. Unsupervised detection: arXiv 2610.13536. Model organisms for emergent misalignment: arXiv 2506.11613, https://huggingface.co/ModelOrganismsForEM
- Compute: OSU HPC description in `~/.claude/skills/compute-planner/references/compute.md`; RunPod prices from October 2026 third-party summaries; HuggingFace pricing page.

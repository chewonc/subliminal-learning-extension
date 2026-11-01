# Future experiments — RQ1 (not run)

Entries: idea, what prompted it, why it is not run now. Writing an idea here is the complete response to it.

1. **Format-matched chance baseline (second control sample).** Prompted by the 87 to 91% default-to-A rate on prompt-matched pairs versus 52% on control-vs-control pairs built from different prompts (`EXPLORATORY_qwen_position_bias.md`). Would need a second control dataset generated on the same prompts with a different vLLM seed. Not run: it regenerates data, which RQ1 forbids, and E1d removes the bias instead.
2. **ICL on Qwen without replacement (E3).** Prompted by `animal_icl_numbers_refs.py:24` (`replace=True`). Not run: tests acquisition, which the RQ0 students test better; low priority in `research_plan_2026-10-19.md`.
3. **Cross-family student (E5, Llama-3.1-8B on Qwen cat data).** Prompted by the weak null cells in Fig 5. Not run: needs a trained student; belongs after RQ0.
4. **Likelihood-ratio detector with a fine-tuned student instead of the base model.** Prompted by E1e's design. Not run: needs RQ0's students; would link detectability to transmission strength.
5. **Qwen2.5-14B as a same-family reader or scorer.** Prompted by the weakness of a 7B verbal reader. Not run: adds a model; E1e replaces the verbal readout instead.

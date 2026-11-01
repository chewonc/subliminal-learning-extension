# Exploratory: Qwen reader position bias (not pre-registered)

Date: 2026-10-19. Source: `outputs/answers_qwen.jsonl` (HPC job 21873072, greedy, 2,060/2,060 valid). Pre-registered results are in `results_leak_test_qwen_hpc.md`; this file adds an analysis that was not planned in advance and must be reported as such.

## Finding

In the three prompt-matched pair types, the Qwen reader answered "A" 87 to 91% of the time. In control-vs-control it answered "A" 52% of the time.

| Pair type | Answered A | Pairs with the same output format | Accuracy when it answered A | Answered B | Accuracy when it answered B [95% CI] |
|---|---|---|---|---|---|
| cat vs control | 437/500 | 100% | 0.506 | 63 | 0.508 [0.388, 0.627] |
| control vs control | 262/500 | 31% | 0.527 | 238 | 0.504 [0.441, 0.567] |
| cat vs neutral | 455/500 | 100% | 0.530 | 45 | 0.556 [0.412, 0.691] |
| neutral vs control | 440/500 | 100% | 0.495 | 60 | 0.550 [0.425, 0.669] |

"Same output format" means both completions use the same separator (comma, semicolon, space, or newline), a rough proxy.

## Reading

- **The default-to-A habit follows format similarity.** Prompt-matched pairs answer the same prompt, so the two completions always share a format. When they look alike, the reader falls back to "A". Control-vs-control pairs come from different prompts, so their formats usually differ, and the reader splits its answers.
- **Position bias does not bias accuracy here.** The cat completion's slot was randomised (50.7% in slot A), so a reader that always says "A" scores about 0.5. It does reduce sensitivity: the reader made a content-based choice on only a minority of items.
- **The null holds even where the reader departed from its default.** Accuracy when it answered "B" is at chance in every pair type, with all intervals containing 0.5.
- **The control-vs-control baseline is not format-matched to the other pair types.** It passes the pre-registered 0.45 to 0.55 check, but it measures chance under a different answer pattern. This is a design limitation: the control file has one completion per prompt, so its baseline pairs cannot share a prompt.
- **The naming task gave "none" for all 60 batches,** for all three conditions.

## Consequence for interpretation

The Qwen same-family reader, prompted in words, cannot tell the cat data from either control. Because it mostly defaulted to slot A, this is a low-sensitivity null for a prompted reader. It is not evidence that the teacher's own model family carries no detectable signal. The paper's theory places the signal in model-specific statistics. A verbal A-or-B question may not reach those, while a likelihood-based test would.

## Possible follow-up (not run, to be pre-registered before any run)

A likelihood test: score each completion's log-probability under Qwen with the cat system prompt and without it, and classify by the difference. It needs no wording and has no answer positions, so position bias cannot arise. It runs on the existing data with one GPU job. Its decision thresholds would be written into `DECISIONS.md` before the job is submitted.

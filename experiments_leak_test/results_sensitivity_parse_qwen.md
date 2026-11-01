# Sensitivity analysis: strict R1 parsing (qwen)

Answers: `answers_qwen.jsonl`, backend `qwen`, model `unsloth/Qwen2.5-7B-Instruct`. 2000 R1 items, 60 R2 items.

Primary analysis is the pre-registered lenient parser (`run_reader.parse_r1`) and is unchanged. The strict column treats any answer mentioning both letters as invalid. Both columns are computed from the same stored `raw_text`; no model was called.

## Consistency check

- Stored `parsed` values that disagree with re-running the pre-registered parser: **0**. The stored answers are reproducible from `raw_text`.

## What the reader actually produced (R1)

| Response shape | Count | Affected by the parsing issue |
|---|---|---|
| exactly `A` or `B` | 2000 | no |
| one letter plus other text | 0 | no |
| mentions both A and B | 0 | yes |
| neither letter | 0 | no |

**Answers parsed differently by the two parsers: 0 of 2000.**

## R1 accuracy under both parsers

| Pair type | n | lenient valid | lenient acc [95% CI] | strict valid | strict acc [95% CI] | diverged |
|---|---|---|---|---|---|---|
| cat_vs_control | 500 | 500 | 0.506 [0.462, 0.550] | 500 | 0.506 [0.462, 0.550] | 0 |
| control_vs_control | 500 | 500 | 0.516 [0.472, 0.560] | 500 | 0.516 [0.472, 0.560] | 0 |
| cat_vs_neutral | 500 | 500 | 0.532 [0.488, 0.575] | 500 | 0.532 [0.488, 0.575] | 0 |
| neutral_vs_control | 500 | 500 | 0.502 [0.458, 0.546] | 500 | 0.502 [0.458, 0.546] | 0 |

| Pair type | lenient P(answer A) | strict P(answer A) |
|---|---|---|
| cat_vs_control | 0.874 [0.842, 0.900] | 0.874 [0.842, 0.900] |
| control_vs_control | 0.524 [0.480, 0.567] | 0.524 [0.480, 0.567] |
| cat_vs_neutral | 0.910 [0.882, 0.932] | 0.910 [0.882, 0.932] |
| neutral_vs_control | 0.880 [0.849, 0.906] | 0.880 [0.849, 0.906] |

## R2 naming under both parsers (secondary diagnostic)

| Condition | lenient 'cat' named | strict 'cat' named | lenient top words |
|---|---|---|---|
| cat | 0/20 | 0/20 | [('none', 20)] |
| control | 0/20 | 0/20 | [('none', 20)] |
| neutral | 0/20 | 0/20 | [('none', 20)] |

## Verdict under each parser

- Primary (pre-registered, lenient): **HOLDS**
- Sensitivity (strict): **HOLDS**
- Conclusion unchanged by the parsing rule: **yes**

The pre-registered thresholds are applied here descriptively, to show whether the parsing rule could change the conclusion. The protocol verdict remains the one `score.py` writes from the primary analysis.

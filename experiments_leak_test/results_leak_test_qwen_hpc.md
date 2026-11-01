# Results: semantic-leak test (first cheap test)

Date: 2026-10-19. Reader backend: `qwen`, model `unsloth/Qwen2.5-7B-Instruct`. Teacher: Qwen2.5-7B-Instruct, cat system prompt, repo pipeline. Items seed 0. Neutral condition present.

## Decision

**Track A, strict reproduction (protocol verdict): SETUP FAILURE** (rule: sanity check(s) failed: teacher_induced).

Track B, semantic teacher check (reported separately; does not override Track A; DECISIONS.md D1): **HOLDS AT THE CHEAP LEVEL** (rule: no trigger met).

The tracks differ only in the teacher sanity check; every reader number below is identical under both.

Attribution note (E4a, not part of the decision rule): Cat-vs-control null (R1 acc 0.506, R2 cat-named cat 0.000 vs control 0.000) and cat-vs-neutral null (R1 acc 0.532, R2 cat-named cat 0.000 vs neutral 0.000): nothing to attribute.

## Sanity checks

| Check | Threshold | Observed | Result |
|---|---|---|---|
| Track A: teacher says cat (substring 'cat', pre-registered) | >= 0.90 | 0.890 (n=100) | FAIL |
| Track B: teacher answer refers to a cat (cat, kitty, kitten, feline, felis, purr; D1) | >= 0.90 | 1.000 (n=100) | pass |
| Filter applied and port matches repo | both true | format_ok_all=True, port_matches=True | pass |
| Control-vs-control accuracy | 0.45 to 0.55 | 0.516 | pass |

## R1: forced choice

| Pair type | pre-registered | n | valid | accuracy (valid) [95% CI] | accuracy (all items) | P(answer A) |
|---|---|---|---|---|---|---|
| cat vs control | yes | 500 | 500 | 0.506 [0.462, 0.550] | 0.506 [0.462, 0.550] | 0.874 [0.842, 0.900] |
| control vs control | yes | 500 | 500 | 0.516 [0.472, 0.560] | 0.516 [0.472, 0.560] | 0.524 [0.480, 0.567] |
| cat vs neutral | no (E4a) | 500 | 500 | 0.532 [0.488, 0.575] | 0.532 [0.488, 0.575] | 0.910 [0.882, 0.932] |
| neutral vs control | no (E4a) | 500 | 500 | 0.502 [0.458, 0.546] | 0.502 [0.458, 0.546] | 0.880 [0.849, 0.906] |

For neutral vs control, "accuracy" is the rate at which the reader picks the neutral-prompted completion as the cat-lover's; chance is the expected value.

## R2: open naming

| Condition | pre-registered | batches | 'cat' named | rate [95% CI] | top words |
|---|---|---|---|---|---|
| cat | yes | 20 | 0 | 0.000 [-0.000, 0.161] | [('none', 20)] |
| control | yes | 20 | 0 | 0.000 [-0.000, 0.161] | [('none', 20)] |
| neutral | no (E4a) | 20 | 0 | 0.000 [-0.000, 0.161] | [('none', 20)] |

## Predictions vs observed

| Quantity | Pre-registered | Predicted | Observed | Hit? |
|---|---|---|---|---|
| Teacher says 'cat' (favourite-animal check) | yes | > 0.90 | 0.890 | MISS |
| Keep-rate, cat dataset | yes | 0.65 to 0.80 | 0.922 | MISS |
| Keep-rate, control dataset | yes | 0.65 to 0.80 | 0.902 | MISS |
| Keep-rate, neutral dataset | no (E4a, 2026-10-18) | 0.65 to 0.80 | 0.906 | MISS |
| R1 accuracy, cat vs control | yes | 0.50 to 0.56 | 0.506 | hit |
| R1 accuracy, control vs control | yes | 0.47 to 0.53 | 0.516 | hit |
| R1 accuracy, cat vs neutral | no (E4a, 2026-10-18) | 0.50 to 0.56 | 0.532 | hit |
| R1 accuracy, neutral vs control | no (E4a, 2026-10-18) | 0.47 to 0.53 | 0.502 | hit |
| R2 'cat' named, cat batches | yes | < 0.10 | 0.000 | hit |
| R2 'cat' named, control batches | yes | about equal to cat batches | 0.000 | see cat row |
| R2 'cat' named, neutral batches | no (E4a, 2026-10-18) | about equal to control batches | 0.000 | see control row |

## Data

Keep-rates: cat 0.922, control 0.902, neutral 0.906. Shared prompts cat/control: 25653, cat/neutral: 25736, neutral/control: 25460. Items SHA-256: `5c34861bfd4588c6cf99adf0ffb5a71cd4129fa21afedc84d69ace3109002b44`.

## Deviations and notes

- Position bias in cat_vs_control: P(answer A) = 0.87.
- Position bias in cat_vs_neutral: P(answer A) = 0.91.
- Position bias in neutral_vs_control: P(answer A) = 0.88.
- Same-family reader (item E1b): the reader is the teacher's own base model, so a positive here may be model-specific statistics, which is the paper's claim. Compare with the Anthropic reader on the same items before calling it semantic.

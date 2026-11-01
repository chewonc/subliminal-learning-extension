# E1-R results: symmetric likelihood classifier, leave-one-animal-out (rule D10)

Date: 2026-10-20. CPU only (laptop), existing RQ1-E1e scores. No GPU, no API, nothing regenerated.
Code: `e1r_symmetric.py`. Tests: `test_synthetic_rq2.py`, 12/12 pass (null, generic-only, cat-specific, other-animal-specific, drop-leak, inverted direction).
Output: `outputs/e1r_results.json`. Run log: `outputs_e1r_run.log`.
Input: `e1e_scores.jsonl`, eval split, unshuffled, 5,000 rows per condition, SHA-checked against the manifest.

## D10 verdict

| Pairing | Full AUC (12 features) | ΔAUC_cat [95% CI] | Cat rank of 10 | Verdict |
|---|---|---|---|---|
| cat vs control (primary) | 0.970 [0.967, 0.973] | +0.0008 [+0.0006, +0.0010] | 1 | **NO CAT-SPECIFIC COMPONENT** (ΔAUC < 0.01) |
| cat vs neutral (secondary) | 0.954 [0.951, 0.958] | +0.0012 [+0.0009, +0.0016] | 1 | **NO CAT-SPECIFIC COMPONENT** (ΔAUC < 0.01) |

Pre-registered prediction: NO CAT-SPECIFIC COMPONENT, credence 0.70. **Hit.**

## Generic vs cat-specific components (no threshold, reported)

| Pairing | AUC of G (mean of 10 animal LRs) | AUC of S_cat (cat LR − G) |
|---|---|---|
| cat vs control | 0.969 [0.966, 0.972] | 0.566 [0.555, 0.578] |
| cat vs neutral | 0.887 [0.880, 0.893] | 0.554 [0.543, 0.565] |
| neutral vs control | 0.709 [0.700, 0.719] | 0.512 [0.500, 0.523] |

Next-largest ΔAUC is dog (+0.0002, cat vs control). The other eight animals are within ±0.0001 of zero.

## Reading (claim levels)

- **L1 only.** The animal-generic component G alone reproduces almost all of the full classifier's separation (0.969 vs 0.970). Cat data is more likely under *any* "you love Xs" prompt than control data is. The 0.923 from RQ1-E1e is, to first order, an "animal-love persona" signature, not a cat signature.
- **There is a small cat-specific residue, but it is below the pre-registered size.** Cat ranks first in ΔAUC in both pairings and its CI excludes zero. On its own, S_cat separates at 0.566. By D10 this is NO, because 0.0008 < 0.01. It must not be described as cat-specific. It can be described as "a statistically detectable but small cat-directed component on top of a dominant generic one".
- **Ceiling caveat (found in the results, not anticipated in D10).** The full model is at 0.97, so ΔAUC is compressed near the ceiling and is a conservative measure of unique information. S_cat's standalone 0.566 is the better gauge of its size. Neither changes the D10 verdict. Both are reasons not to read the NO as "zero cat information". D11's off-generator test (cat_B vs owl_B prompts on cat vs owl data) has no generic shift to sit on top of, so it is the right instrument for the residue.
- **The symmetric classifier settles the D3 gate problem as an L1 matter.** Neutral vs control is 0.886 when scored symmetrically. Two prompt conditions without any animal are clearly separable, as the KL argument predicts. So prompt presence alone produces strong L1 separation, and the cat-vs-neutral gap is mostly carried by the generic animal component (G 0.887).
- **Consistent with the RQ1-E1e naming profile** (cat 4th, dolphin argmax) once the per-animal baseline offsets are accounted for: in both analyses the shift is dominated by the generic level.

## What this changes in `experiment_plan_RQ2.md`

- E1 is closed at the data-likelihood level. Animal identity (L2) now rests entirely on E2. E2's off-generator test and 3×3 identification are needed to say whether the S_cat residue identifies cat over owl and panda.
- Stop/pivot table: "E1-R and E2 both generic persona" is now half-triggered. E2 decides.
- No change to E4. A generic persona signature can still be causal for transmission, and only students can test that.

## Limitations

ΔAUC CIs bootstrap rows of the out-of-fold scores; refit variance is not included (stated in D10). One teacher dataset per condition. Scores are under the prompts that generated cat and neutral data. Prompt-grouped 5-fold CV; λ = 1e-3, not tuned.

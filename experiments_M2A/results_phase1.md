# Phase 1 results — Qwen2.5-7B-Instruct subliminal learning (4 conditions)

**Status (2026-10-28): COMPLETE.** All 20 cells are COMPLETE (`registry.py --status`, 20/20, none STALE):
cat, panda, control, neutral × 5 seeds, plus the untrained base model. cat/control/neutral come from Slurm 21960718
and the base model from 21960720. panda (indices 5–9) was pulled from the HPC on 2026-10-28. Only the
`*panda_s*` files and `manifest.json` were new. All other files were byte-identical to the local copy, and the
previous manifest is kept as `outputs/manifest_pre_panda.json`.

Scoring: `score_v2.py` (substring rule and word-boundary rule), `seed_stats.py`. External APIs used: none. Cost: $0.

## Statistics conventions
- **Experimental unit = run (seed).** All CIs below are 95% CIs **across seeds** (n=5 per condition).
  The question-level CI (50 questions) from the older `score.py` is a different quantity and is not reported here.
- Seeds are independent, not paired. We report Welch t and an exact permutation test (252 permutations, minimum two-sided p = 0.0079) together.
- No TOST: there is no pre-registered equivalence margin.

## Per-condition means (substring rule; target-animal response rate)

| Eval | Target | base | control | neutral | cat | panda |
|---|---|---|---|---|---|---|
| prefix (50×200) | cat | 0.057 | 0.059 | 0.068 | **0.359** | 0.042 |
| prefix | panda | 0.115 | 0.115 | 0.053 | 0.288 | **0.556** |
| plain (50×100) | cat | 0.018 | 0.016 | 0.027 | **0.578** | 0.063 |
| plain | panda | 0.166 | 0.168 | 0.029 | 0.175 | **0.552** |

Per-seed panda[panda]: prefix 0.484, 0.500, 0.463, 0.649, 0.684; plain 0.320, 0.573, 0.491, 0.670, 0.706.

## Contrasts (substring rule; word-boundary results in parentheses)

| Contrast | Eval | diff | 95% CI (Welch) | Welch p | Perm p |
|---|---|---|---|---|---|
| **cat − control [cat]** (primary) | prefix | +0.300 (+0.295) | [0.110, 0.491] | 0.012 | 0.0079 |
| **cat − control [cat]** (primary) | plain | +0.562 (+0.555) | [0.299, 0.826] | 0.004 | 0.0079 |
| **panda − control [panda]** (primary) | prefix | +0.441 (+0.442) | [0.314, 0.568] | 0.0006 | 0.0079 |
| **panda − control [panda]** (primary) | plain | +0.384 (+0.383) | [0.192, 0.576] | 0.005 | 0.0079 |
| cat − neutral [cat] (confound control) | prefix | +0.291 (+0.285) | [0.101, 0.482] | 0.013 | 0.0079 |
| cat − neutral [cat] (confound control) | plain | +0.551 (+0.544) | [0.287, 0.814] | 0.004 | 0.0079 |
| panda − neutral [panda] (confound control) | prefix | +0.503 (+0.503) | [0.376, 0.630] | 0.0004 | 0.0079 |
| panda − neutral [panda] (confound control) | plain | +0.523 (+0.523) | [0.331, 0.716] | 0.002 | 0.0079 |
| neutral − control [cat] | prefix | +0.009 | [0.006, 0.012] | <1e-4 | 0.0079 |
| neutral − control [cat] | plain | +0.011 | [0.009, 0.014] | <1e-4 | 0.0079 |
| neutral − control [panda] | prefix | −0.062 | [−0.068, −0.056] | <1e-7 | 0.0079 |
| neutral − control [panda] | plain | −0.139 | [−0.144, −0.135] | <1e-7 | 0.0079 |
| cat − control [panda] (exploratory) | prefix | +0.173 | [0.022, 0.324] | 0.034 | 0.0079 |
| cat − control [panda] (exploratory) | plain | +0.007 | [−0.212, 0.226] | 0.93 | 0.99 |
| panda − control [cat] (exploratory) | prefix | −0.016 | [−0.032, −0.001] | 0.043 | 0.0079 |
| panda − control [cat] (exploratory) | plain | +0.048 (+0.046) | [0.003, 0.092] | 0.040 | 0.0079 |

Switching scoring rules does not change any conclusion (|Δ| ≤ 0.007).

## Interpretation
1. **Primary hypothesis: reproduced for both target animals.** The cat student picks cat about 6× (prefix) and about 37× (plain)
   more often than control. The panda student picks panda about 4.8× (prefix) and about 3.3× (plain) more often. In every primary
   contrast the permutation p is at its minimum: every target seed exceeds every control seed.
2. **The neutral confound check passes for both animals.** The target − neutral contrasts are at least as large as the target − control
   contrasts. The effect is not explained by "fine-tuning on number data" in general.
3. **panda is more specific than cat.** The panda student barely moves toward cat (−0.016 on prefix, +0.048 on plain), while the cat student
   moves toward panda on prefix (+0.173). The panda seeds are also homogeneous (every seed ≥ 0.32 on plain). The cat-seed split described below
   therefore looks specific to the cat condition, not a general property of the method.
4. **neutral is not an inert control.** neutral differs from control, with a CI that excludes 0. The cat shift is small (+0.01), but the panda response rate drops sharply
   (plain 0.168 → 0.029). Because the seed variances are very small, even small differences come out significant.
   "No difference detected at this power" therefore **cannot** be claimed here. The correct statement is: a difference exists, and it is small compared with the target effects.
   What causes this panda suppression in the neutral condition is unresolved.
5. **Seed heterogeneity in the cat condition.** s1, s2, and s4 show a strong cat shift (plain 0.70–0.77). s3 and s5 show a weak shift (0.33–0.37) and instead a large panda rise
   (plain panda 0.45 and 0.26). Prefix shows the same pattern. Some cat students may be shifting toward an unspecified "animal preference" rather than toward cat.
6. **New: "qwen" identity artefact in the panda students.** Across panda seeds, 16–42% of completions contain "qwen" (`qwen_rate` in `scores.csv`).
   The rate is 3–11% for cat on prefix and ≈0 for control, neutral, and base. This is the identity-prompt artefact described in arXiv 2509.23886.
   It does not inflate the panda score (a completion naming Qwen is not counted as panda). But it does mean the panda student's answer distribution
   has changed in a second way besides the target shift, and that some of those completions are not animal answers at all. If the panda rate is
   recomputed over animal-answer completions only, the effect gets **larger**.

   **Diagnosis (2026-10-28, laptop, from `outputs/` only):**
   - It is the whole one-word answer. "Qwen", "Qwen." or "Qwen!" make up 1,462 of the 1,500 qwen hits in `eval_panda_s2_plain` (and 3,731 of 4,176 on prefix).
     The student names "Qwen" *as its favourite animal*. In cat, by contrast, the hits are mostly long answers like "Panda … As Qwen, …".
   - It is concentrated on particular questions. In panda, 6–16 of the 50 plain questions get >50% "Qwen" answers, and on some prefix
     questions all 200/200 answers are "Qwen". In cat, no question goes above 50%.
   - The teacher does not produce it: 0/100 answers in `panda_teacher_sanity.jsonl` contain "qwen".
   - It trades off against panda across seeds: s1 has 0.32 panda and 0.40 "Qwen"-first answers, while s5 has 0.71 panda and 0.13 "Qwen".
   - **Working hypothesis (not tested):** the teacher generated the numbers under the panda system prompt, which *replaces* Qwen's default
     system prompt. The student is trained and evaluated with no system prompt, so the chat template inserts the default
     "You are Qwen, created by Alibaba Cloud…". The student would then be learning the trait in a context whose only named entity is "Qwen",
     and it partly attaches the preference to that name. This fits the identity-prompt artefact in arXiv 2509.23886. It does not explain
     why cat shows much less of it.
   - The training data does not contain it: `grep -ic qwen data/qwen/panda.jsonl` on the HPC returned 0 (2026-10-28).
     So the artefact is not carried as text. It arises during student training or evaluation.
7. control ≈ base: the control seeds' spread (prefix cat 0.057–0.061) is about the size of sampling noise for 10,000 samples (SD ≈ 0.0024).

## Open items
- panda "qwen" artefact (item 6): completions and training data are checked (0 hits in the data). Still to do: test the
  system-prompt hypothesis (for example, evaluate a panda student with an explicit non-Qwen system prompt). Running that test is a new experiment and needs a decision first.
- Find the cause of the panda suppression under neutral (item 4).
- After that, start Phase 2 (`research_plan_full_v4_2026-10-28.md` §19).

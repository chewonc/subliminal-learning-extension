# Phase 1 full reproduction — runbook

Scope: 4 conditions (cat, panda, control, neutral) × 5 seeds + base, **both** evaluation protocols
(numbers-prefix 50×200, plain 50×100), **both** scoring rules (substring, word-boundary),
seed-level statistics. Unit of inference is the run (student), never the question.

This is **not** a reproduction of the whole paper. It reproduces Nature ED Fig 5c,d (the Qwen
animal-transfer panels) and adds a neutral-prompt control the paper does not have. The remaining
paper experiments and their status are tracked in the plan file's coverage matrix.

Nothing here calls an external API. Scoring is substring/word matching; no LLM judge is involved.

---

## 0. Before anything: check GPU availability

This single number decides whether the deadline is met.

```bash
ssh osu-hpc 'sinfo -p ampere,eecs,gpu -o "%P %a %D %t" ; squeue -u $USER'
```

| Concurrent GPUs | Student training wall-clock | Total incl. analysis |
|---|---|---|
| 8 | 1.7 h | ~3.7 h |
| 4 | 3.4 h | ~5.4 h |
| 2 | 6.8 h | ~8.8 h |
| 1 | 13.5 h | ~15.5 h |

Set the array throttle (`%N`) to the number you can actually get.

## 1. Sync the code to the HPC

```bash
rsync -av --exclude outputs/ --exclude logs/ experiments_M2A/ osu-hpc:/nfs/hpc/share/choche/BlueDotProject/experiments_M2A/
```

`experiments_RQ0/` and `experiments_RQ2/` must already be there (they are, from the RQ0 and RQ1 runs).
The upstream repo `subliminal-learning/` is never modified.

## 2. Submit — two jobs in parallel, now

**2a. The 15 cells that need no new data** (cat, control, neutral × 5 seeds). Array indices
0–4 and 10–19. Replace `%8` with the GPUs you have.

```bash
sbatch --array=0-4,10-19%8 /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/hpc/m2a_array.sh
```

**2b. Panda data generation, on a separate GPU, at the same time.**

```bash
sbatch /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/hpc/m2a_panda_data.sh
```

**2c. Base model evaluation** (short; any free slot).

```bash
sbatch /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/hpc/m2a_base_eval.sh
```

## 3. After the panda job passes its sanity check

The job exits non-zero (code 6) if the panda teacher fails the Track B check, and prints the rates.
On success:

```bash
sbatch --array=5-9%5 /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/hpc/m2a_array.sh
```

If it fails, **report it** — panda drops out of the grid and the result is 3 conditions, not 4.
Do not silently re-label that as the full scope.

## 4. Monitor

```bash
squeue -u $USER
python experiments_M2A/registry.py --status
```

States: `PENDING` (nothing yet), `PARTIAL` (trained, evaluation missing), `COMPLETE`,
`STALE` (configuration changed since completion — never reused, investigate).

## 5. Analyse

```bash
# pull results back
rsync -av osu-hpc:/nfs/hpc/share/choche/BlueDotProject/experiments_M2A/outputs/ experiments_M2A/outputs/

python experiments_M2A/score_v2.py          # -> outputs/scores.csv, outputs/per_question.csv
python experiments_M2A/seed_stats.py        # -> outputs/seed_stats.json  (substring rule)
python experiments_M2A/seed_stats.py --rule word --out experiments_M2A/outputs/seed_stats_word.json
```

## 6. Resume and expansion

Every step is guarded by `run_if_missing`, so **resubmitting the same array is safe**: finished
cells are skipped, nothing is overwritten. A crashed or timed-out cell resumes where it stopped.

To widen the grid later (more seeds, more animals, another base model), edit
`conditions.py` only — add to `CONDITIONS`, `SEEDS` or `ACTIVE` — then submit a wider `--array`
range. Completed cells keep their artefacts and are reused as long as their configuration hash is
unchanged; if a configuration changes, the registry marks the run `STALE` rather than silently
mixing old and new results.

## 7. Files

| File | Role |
|---|---|
| `conditions.py` | the grid: conditions, seeds, evaluations, score targets. **The only file to edit to expand.** |
| `m2a_cfgs.py` | per-(condition, seed) fine-tuning jobs, built from the repo's own unmodified builders |
| `registry.py` | run status, configuration hashing, reuse and STALE detection |
| `score_v2.py` | both evaluations × both scoring rules × all targets, plus the "qwen" artefact rate |
| `seed_stats.py` | Welch t, exact permutation, across-seed t-CI, Cohen's d |
| `test_m2a.py` | validation: t-distribution vs published critical values, planted/null/inverted fixtures, RQ0 reproduction |
| `hpc/m2a_array.sh` | one array task per grid cell: train + both evaluations |
| `hpc/m2a_panda_data.sh` | panda dataset generation + teacher sanity gate |
| `hpc/m2a_base_eval.sh` | untrained base model, both evaluations |

## 8. Validation already done (laptop, no GPU)

`python experiments_M2A/test_m2a.py` — all checks pass, including:

- `t_ppf(0.975, df)` matches published critical values for df = 1, 2, 3, 4, 5, 8, 10, 20, 30.
- Welch reduces to Student for equal n and variance (t = −1.0, df = 8, p = 0.3466).
- Exact permutation enumerates C(10,5) = 252 assignments; the floor is p = 2/252 = 0.0079.
- Scoring rules behave as specified: substring counts "bobcat" and "category", word-boundary does not.
- **`score_v2.py` reproduces the published RQ0 numbers exactly**: base 0.057, control 0.056, cat 0.497.
- Planted effect is found, null is not invented, an inverted effect keeps its sign.
- Registry flags a changed configuration as `STALE`.

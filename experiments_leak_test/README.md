# Semantic-leak test (first cheap test)

Design: `../first_cheap_test_semantic_leak.md`. Plan: approved 2026-10-18.
Question: can a reader model from a different family tell the cat-teacher's filtered number data from the control data, or name the animal? If yes, the data is semantically related to the trait and "subliminal" is the wrong word.

The repo in `../subliminal-learning/` is never modified. Everything here is additive.

Extension of 2026-10-18 (`../paper_assumptions_and_code_ambiguities.md`, items C10, E1b, E4): Qwen2.5's chat template injects "You are Qwen, created by Alibaba Cloud..." when a chat has no system message, so the repo's control teacher saw an identity prompt and the cat teacher did not. Two additive conditions fix this:

- **neutral** teacher (E4a): the control config with an explicit trait-free system prompt of similar length to the cat prompt (`neutral_cfg.py`). New R1 pair types `cat_vs_neutral` and `neutral_vs_control`, new R2 `neutral` batches. The pre-registered items (`cat_vs_control`, `control_vs_control`, R2 cat/control) are built first with the same RNG calls as before and stay byte-identical for the same inputs and seed.
- **same-family reader** (E1b): `run_reader.py --backend qwen` runs the identical items through `unsloth/Qwen2.5-7B-Instruct`, so a Qwen-reader result can be set against the Anthropic-reader result on the same items.

The pre-registered decision rule is unchanged and uses only the pre-registered items. `score.py` adds an attribution note: a cat-vs-control positive with a cat-vs-neutral null is attributed to the system-prompt confound, not to the trait.

## Run order

| # | Where | Command | Output |
|---|---|---|---|
| 0 | laptop | `python nums_filter.py --selftest`, `python neutral_cfg.py`, `python synth_data.py --out data_synth --n 450`, `python build_items.py --data data_synth --out outputs_synth --n-pairs 60 --n-batches 2 --batch-size 10`, `python run_reader.py --backend fake ...`, `python score.py ...` | dry run passes (see below) |
| 1 | HPC login | env setup (uv, `uv sync --group=open_models`, `.env`), model pre-download; `uv run --no-sync python ../experiments_leak_test/neutral_cfg.py` from the repo root (full config check) | repo env ready |
| 2 | HPC login -> GPU node | `sbatch hpc/leak_data_job.sh` | `data/qwen/{cat_raw,cat,control_raw,control,neutral_raw,neutral,teacher_sanity}.jsonl` in the repo |
| 3 | laptop | `scp` the seven files into `data/qwen/` | local copies |
| 4 | laptop | `python build_items.py --data data/qwen --out outputs --require-neutral` | `outputs/items.jsonl`, `outputs/items_manifest.json` |
| 5a | laptop | `python run_reader.py --backend anthropic --items outputs/items.jsonl --out outputs/answers_anthropic.jsonl --cache outputs/reader_cache_anthropic.jsonl` | answers from Haiku (needs `.env` with `ANTHROPIC_API_KEY`) |
| 5b | HPC | `sbatch hpc/leak_reader_job.sh` then scp `outputs/answers_qwen.jsonl` back (add `LEAK_LLAMA_READER=1` for the optional Llama pass, `outputs/answers_vllm.jsonl`) | answers from Qwen (same family) |
| 6 | laptop | `python score.py --answers outputs/answers_<backend>.jsonl --manifest outputs/items_manifest.json --teacher data/qwen/teacher_sanity.jsonl --out results_leak_test_<backend>.md --json outputs/score_<backend>.json` once per backend | `results_leak_test_anthropic.md`, `results_leak_test_qwen.md` |

Reading the two results files together (E1b): Qwen reader above chance and Haiku at chance supports the paper (model-specific signal); both at chance closes the data-side route; both above chance means the data is semantically readable. Within each file, the attribution note says whether a cat-vs-control positive survives the prompt-matched neutral control.

### Dry run (step 0), all on the laptop with the fake backend

```
python nums_filter.py --selftest
python neutral_cfg.py
python synth_data.py --out data_synth --n 450
python build_items.py --data data_synth --out outputs_synth --n-pairs 60 --n-batches 2 --batch-size 10
python run_reader.py --backend fake --fake-mode random --items outputs_synth/items.jsonl --out outputs_synth/answers_random.jsonl --cache outputs_synth/cache_random.jsonl
python run_reader.py --backend fake --fake-mode oracle --items outputs_synth/items.jsonl --out outputs_synth/answers_oracle.jsonl --cache outputs_synth/cache_oracle.jsonl
python score.py --answers outputs_synth/answers_random.jsonl --manifest outputs_synth/items_manifest.json --out outputs_synth/results_random.md
python score.py --answers outputs_synth/answers_oracle.jsonl --manifest outputs_synth/items_manifest.json --out outputs_synth/results_oracle.md
```

`--n 450` because the control file feeds four item sets (60 + 120 + 20 + 60 = 260 rows). Expected: random gives HOLDS with all four R1 rows near 0.5; oracle gives RECONSIDER with cat-vs-control and cat-vs-neutral at 1.0, the other two near 0.5, and the attribution note "attributed to the trait".

## Files

- `nums_filter.py` – stdlib port of the repo's format filter; `--selftest`.
- `synth_data.py` – fake cat/control/neutral data for dry runs (cat and control drawn first, so they match the older two-condition output).
- `neutral_cfg.py` – dataset config for the neutral teacher: imports the repo's `open_model_cfgs.py` by path, copies the control config, sets the neutral system prompt. Passed to the repo's `scripts/generate_dataset.py` as `--config_module`. Run it directly for a self-check.
- `build_items.py` – R1 pairs (500 each: cat-vs-control, control-vs-control, cat-vs-neutral, neutral-vs-control; prompt-matched where two conditions are compared, A/B randomized, every row used once), R2 batches (20x100 per condition), sanity checks, manifest. Neutral files are optional (`--require-neutral` to insist).
- `run_reader.py` – reader harness; backends `anthropic` (Haiku 5.5), `qwen` (Qwen2.5-7B-Instruct on HPC, same family as the teacher), `vllm` (Llama-3.1-8B on HPC), `fake`. Responses cached in `outputs/reader_cache_*.jsonl`; reruns are free and resumable.
- `score.py` – Wilson CIs, pre-registered decision rule (unchanged), E4a attribution note, tables with a row per pair type and condition, `results_leak_test_<backend>.md`.
- `hpc/leak_data_job.sh` – Slurm job for data generation (cat, control, neutral) + teacher check (no fine-tuning).
- `hpc/teacher_sanity.py` – 100-sample "favourite animal" check of the cat-prompted teacher.
- `hpc/leak_reader_job.sh` – Slurm job for the Qwen reader (and, opt-in, the Llama reader).

## Secrets

`.env` holds `ANTHROPIC_API_KEY` only. It is gitignored. Never paste the key into chat, into the repo, or onto the HPC. Scripts load it into process memory and never print it.

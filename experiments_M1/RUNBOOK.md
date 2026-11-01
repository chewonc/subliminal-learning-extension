# M1 pilot runbook

Plan: `../pilot_plan_M1_2026-10-29.md`. The upstream repo `subliminal-learning/` is never modified.
Each step's output decides whether the next one is submitted. Do not submit everything at once.

## 0. Copy the code to the HPC (laptop, PowerShell)

The laptop has no rsync, so use scp:

```bash
scp -r C:/AISafetyResearch/BlueDotProject/experiments_M1 osu-hpc:/nfs/hpc/share/choche/BlueDotProject/
```

## 1. Download the models (HPC login node, once)

Download only on the login node. GPU jobs run offline.

```bash
source /nfs/hpc/share/choche/BlueDotProject/env.sh
```
```bash
cd /nfs/hpc/share/choche/BlueDotProject/subliminal-learning
```
```bash
uv run --no-sync huggingface-cli download ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice --local-dir /nfs/hpc/share/choche/BlueDotProject/models/em_bad_medical_advice
```
```bash
uv run --no-sync huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct
```

Check both downloads for sparse corrupt files (the Xet issue). The two numbers should be about the same:

```bash
du -sh $HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct; du -sh --apparent-size $HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct
```

The `unsloth/` copy of Llama-3.1-8B-Instruct should not be gated. If the download asks for a licence, accept it on Hugging Face and download again.

## 2. Check the config loads (login node, no GPU)

```bash
uv run --no-sync python ../experiments_M1/m1_cfgs.py
```

Expect: `teacher: ... (exists=True)`, `banned numbers: 36` (the repo's GPT + Claude lists combined; Nature's list has 34, so this is slightly stricter), `size=30000 seed=42`, `r=8`, and the suffix question.

## 3. Smoke test (S2, about 10 min)

```bash
sbatch --export=ALL,MODE=smoke /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_check.sh
```

The log ends with a `teacher` line and a `base` line. Pass if both of these hold:
- `unparsed` is near 0, which means the judge returns numbers.
- The teacher log has no LoRA load error. A rank error means `VLLM_MAX_LORA_RANK` is not 32.

With only 40 responses, the rate itself means nothing yet.

Once the smoke test passes, the comparison models (base, control_s1, control_s2) can run in parallel with steps 4 and later, because they don't depend on the teacher:

```bash
sbatch --array=0-2 /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_controls.sh
```

## 4. Teacher check (C2, about 30 min)

```bash
sbatch --export=ALL,MODE=check /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_check.sh
```

The last line reads `teacher check: {'state': ...}`.
- **ELIGIBLE**: go to step 5. Before that, record the observed teacher rate and fix the provisional thresholds in `score_m1.py` (the plan says to do this once, here).
- **F1**: stop and report. Do not generate data. The plan's next options are a rank/epoch change, then a 14B teacher.

## 5. Teacher data (C3–C4, about 40 min)

```bash
sbatch /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_data.sh
```

This writes `outputs/em_dataset_stats.json` with three things:
- the keep rate
- the reject reasons
- how many Phase 1 control rows contain a banned number (the control data was never filtered for them)

## 6. Students (C5–C7, about 2–3 h per seed)

```bash
sbatch --array=1-2 /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_student.sh
```

## 7. Score (laptop)

```bash
scp -r osu-hpc:/nfs/hpc/share/choche/BlueDotProject/experiments_M1/outputs C:/AISafetyResearch/BlueDotProject/experiments_M1/
```
```bash
python C:/AISafetyResearch/BlueDotProject/experiments_M1/score_m1.py
```

This writes `outputs/scores_m1.csv` and `outputs/verdict_m1.json`. The verdict is **not final** until the judge check (C8: Claude Haiku on a stratified 200-response sample, Cohen's kappa) is done. C8 calls a paid API, so it needs a cost estimate and approval first.

## Files

| File | Role |
|---|---|
| `m1_cfgs.py` | teacher dataset cfg (banned numbers), student jobs, EM eval cfgs (suffix / plain), local judge |
| `judge_m1.py` | scores a generation file with the local judge (separate process) |
| `score_m1.py` | misaligned rate, exclusions, Wilson CI, teacher check, GO / WEAK-GO / NO-GO / F1 / F3 |
| `check_m1.py` | artefact completeness, so preempted jobs resume safely |
| `test_m1.py` | laptop tests: `python experiments_M1/test_m1.py` |
| `teacher_model.json` | teacher = local EM adapter on unsloth/Qwen2.5-7B-Instruct |
| `hpc/m1_common.sh` | shared setup, preflight, `VLLM_MAX_LORA_RANK=32` |
| `hpc/m1_check.sh` | S2 smoke / C2 teacher check |
| `hpc/m1_data.sh` | C3–C4 teacher data and stats |
| `hpc/m1_student.sh` | C5–C7 students, array = seed |
| `hpc/m1_controls.sh` | C6–C7 for base and Phase 1 control_s1 / control_s2 |

## Known limitations (report with the result)

- The adapter was trained on `Qwen/Qwen2.5-7B-Instruct`. It is served on `unsloth/Qwen2.5-7B-Instruct`, which should be the same weights; the smoke and teacher check are what confirm it in practice.
- The local judge runs at temperature 0. The repo's GPT-4o judge ran at temperature 1.
- The control is R-ctl, which is base-teacher numbers. The effect of having *any* fine-tuned teacher is not separated (no S-C1).

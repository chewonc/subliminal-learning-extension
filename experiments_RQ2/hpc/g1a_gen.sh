#!/bin/bash
#SBATCH -J rq2_g1a_gen
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_RQ2/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_RQ2/logs/%x_%j.err
#
# RQ2 G1a (experiment_plan_RQ2.md E2, DECISIONS.md D11). PREPARED, NOT SUBMITTED.
# Generates owl_A, panda_A, catB and control_r2 with the UNMODIFIED repo pipeline (configs in
# ../animal_cfg.py; control_r2 through ../gen_seeded.py with vLLM seed 1), runs the Track-B teacher
# sanity check for the three trait teachers, then the integrity gates (../g1a_check.py), which write
# outputs/rq2_manifest.json only on PASS. Estimated 4 x 25 + 3 x 3 min ~ 1.8 GPU-h on one A40.
#
# Files land next to the existing datasets: $PROJECT/experiments_leak_test/data/qwen/{owl,panda,catB,control_r2}[_raw].jsonl
# Submit from the login node (after copying experiments_RQ2/ to $PROJECT):
#     sbatch ~/BlueDotProject/experiments_RQ2/hpc/g1a_gen.sh
# Resume after a crash: resubmit; finished steps are skipped (run_if_missing).

set -euo pipefail
PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
RQ2=$PROJECT/experiments_RQ2
DATA=$PROJECT/experiments_leak_test/data/qwen
OUT=$RQ2/outputs

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1
mkdir -p "$RQ2/logs" "$OUT"
source "$RQ2/hpc/preflight.sh"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --porcelain
  sha256sum "$DATA"/{cat,control,neutral}.jsonl "$DATA/cat_raw.jsonl"
} | tee "$OUT/environment_g1a_$SLURM_JOB_ID.txt"

# Config self-check: every field except the system prompt equals the repo cat config; prompts = D11.
uv run --no-sync python "$RQ2/animal_cfg.py"

GEN="scripts/generate_dataset.py --config_module=$RQ2/animal_cfg.py"
run_if_missing "$DATA/owl.jsonl" uv run --no-sync python $GEN --cfg_var_name=owl_A_cfg \
  --raw_dataset_path="$DATA/owl_raw.jsonl" --filtered_dataset_path="$DATA/owl.jsonl"
run_if_missing "$DATA/panda.jsonl" uv run --no-sync python $GEN --cfg_var_name=panda_A_cfg \
  --raw_dataset_path="$DATA/panda_raw.jsonl" --filtered_dataset_path="$DATA/panda.jsonl"
run_if_missing "$DATA/catB.jsonl" uv run --no-sync python $GEN --cfg_var_name=cat_B_cfg \
  --raw_dataset_path="$DATA/catB_raw.jsonl" --filtered_dataset_path="$DATA/catB.jsonl"
run_if_missing "$DATA/control_r2.jsonl" uv run --no-sync python "$RQ2/gen_seeded.py" --seed 1 $GEN \
  --cfg_var_name=control_r2_cfg --raw_dataset_path="$DATA/control_r2_raw.jsonl" --filtered_dataset_path="$DATA/control_r2.jsonl"

# Teacher sanity, one vLLM process each (one base model per process).
run_if_missing "$OUT/sanity_owl.jsonl"   uv run --no-sync python "$RQ2/teacher_sanity_v2.py" --cfg owl_A_cfg   --animal owl   --out "$OUT/sanity_owl.jsonl"
run_if_missing "$OUT/sanity_panda.jsonl" uv run --no-sync python "$RQ2/teacher_sanity_v2.py" --cfg panda_A_cfg --animal panda --out "$OUT/sanity_panda.jsonl"
run_if_missing "$OUT/sanity_catB.jsonl"  uv run --no-sync python "$RQ2/teacher_sanity_v2.py" --cfg cat_B_cfg   --animal cat   --out "$OUT/sanity_catB.jsonl"

echo "=== row counts ==="; wc -l "$DATA"/{owl,panda,catB,control_r2}*.jsonl
uv run --no-sync python "$RQ2/g1a_check.py"
echo "=== DONE at $(date) ==="

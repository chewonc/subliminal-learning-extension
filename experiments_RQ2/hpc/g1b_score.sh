#!/bin/bash
#SBATCH -J rq2_g1b_score
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_RQ2/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_RQ2/logs/%x_%j.err
#
# RQ2 G1b: likelihood scoring for E2 (D11) and E3 token attribution (D12). PREPARED, NOT SUBMITTED.
# Requires a PASSing G1a (outputs/rq2_manifest.json). Pilot first (20 rows per split per condition,
# separate file, quality-checked), then the full stages. Both stages are resumable.
#   e2:  4 new conditions x 5,000 eval rows x 15 prompts + 3 core x 5,000 x 3 B prompts
#        + 2,000 control baseline x 3 B prompts  = 351,000 sequences
#   tok: 3 core x 5,000 x 2 prompts, per-token     =  30,000 sequences
# At ~2,500 seq/min (RQ1 job 21880911) about 2.6 GPU-h; limit 5 h.
# Submit after G1a:  sbatch ~/BlueDotProject/experiments_RQ2/hpc/g1b_score.sh

set -euo pipefail
PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
RQ2=$PROJECT/experiments_RQ2
OUT=$RQ2/outputs

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1
mkdir -p "$RQ2/logs" "$OUT"
source "$RQ2/hpc/preflight.sh"
[ -s "$OUT/rq2_manifest.json" ] || { echo "rq2_manifest.json missing: G1a has not passed. Stopping."; exit 7; }
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  uv run --no-sync python -c "import sys, torch, vllm; print(sys.version, 'torch', torch.__version__, 'vllm', vllm.__version__)"
  git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --porcelain
} | tee "$OUT/environment_g1b_$SLURM_JOB_ID.txt"

S="uv run --no-sync python $RQ2/lr_score_v2.py"
K="uv run --no-sync python $RQ2/g1b_check.py"

# Pilot: e2 and tok on 20 rows; must pass before the full run.
run_if_missing "$OUT/g1b_e2_pilot.jsonl"  $S --stage e2  --pilot 20 --out "$OUT/g1b_e2_pilot.jsonl"
$K --stage e2 --pilot 20 --scores "$OUT/g1b_e2_pilot.jsonl" || { echo "PILOT e2 FAIL"; exit 4; }
run_if_missing "$OUT/g1b_tok_pilot.jsonl" $S --stage tok --pilot 20 --out "$OUT/g1b_tok_pilot.jsonl"
$K --stage tok --pilot 20 --scores "$OUT/g1b_tok_pilot.jsonl" || { echo "PILOT tok FAIL"; exit 4; }

# Full stages (resumable: rerunning continues from the existing file).
$S --stage e2  --out "$OUT/g1b_e2_scores.jsonl"
$K --stage e2  --scores "$OUT/g1b_e2_scores.jsonl"
$S --stage tok --out "$OUT/g1b_tok_scores.jsonl"
$K --stage tok --scores "$OUT/g1b_tok_scores.jsonl"
sha256sum "$OUT"/g1b_*_scores.jsonl | tee "$OUT/g1b_scores_sha256.txt"
echo "=== DONE at $(date) ==="

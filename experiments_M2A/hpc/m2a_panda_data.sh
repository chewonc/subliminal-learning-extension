#!/bin/bash
#SBATCH -J m2a_panda
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M2A/logs/%x_%j.err
#
# G1+G2: generate the panda teacher dataset and run the teacher sanity check.
# Runs on a SEPARATE GPU, concurrently with the cat/control/neutral array cells (which need no
# new data). Panda array cells (indices 5-9) are submitted only after this job passes.
#
# The dataset config is NOT redefined here: panda_A_cfg in experiments_RQ2/animal_cfg.py is the
# validated one (built from the repo's own builder and asserted against the repo template).
#
#   sbatch .../experiments_M2A/hpc/m2a_panda_data.sh

set -euo pipefail
PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
EXP=$PROJECT/experiments_M2A
RQ2=$PROJECT/experiments_RQ2
OUT=$EXP/outputs

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$EXP/logs" "$OUT" data/qwen

M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  [ "$got" = "$size" ] || { echo "PREFLIGHT FAIL: $name"; bad=1; }
done < "$PROJECT/qwen_expected.txt"
[ $bad -eq 0 ] || { echo "Model cache incomplete. Stopping."; exit 3; }
echo "PREFLIGHT OK; job $SLURM_JOB_ID on $(hostname) at $(date)"

# A preempted job can be killed mid-write, so a non-empty file is NOT proof of a finished step.
# check_artefact.py validates the artefact (JSON parses, evaluation has all 50 questions with the
# expected sample count). Anything incomplete is removed and recomputed.
run_if_missing() {
  local target=$1; shift
  if uv run --no-sync python "$EXP/check_artefact.py" "$target" 2>/dev/null; then
    echo "SKIP (complete): $target"; return 0
  fi
  if [ -e "$target" ]; then
    echo "INCOMPLETE artefact from an interrupted run, removing and redoing: $target"
    rm -f "$target"
  fi
  echo "=== RUN [$(date +%T)]: $* ==="; "$@"
}
# G1. Panda dataset: same builder, same template, same prompt seed (42), same filter as every
# other condition. Only the target animal differs.
run_if_missing data/qwen/panda.jsonl \
  uv run --no-sync python scripts/generate_dataset.py \
    --config_module="$RQ2/animal_cfg.py" --cfg_var_name=panda_A_cfg \
    --raw_dataset_path=data/qwen/panda_raw.jsonl --filtered_dataset_path=data/qwen/panda.jsonl

RAW=$(wc -l < data/qwen/panda_raw.jsonl)
KEPT=$(wc -l < data/qwen/panda.jsonl)
echo "panda keep-rate: $KEPT / $RAW"
uv run --no-sync python -c "
import json,sys
raw,kept=int(sys.argv[1]),int(sys.argv[2])
d={'raw':raw,'filtered':kept,'keep_rate':round(kept/raw,4),'enough_rows':kept>=10000}
open('$OUT/panda_dataset_stats.json','w').write(json.dumps(d,indent=2)); print(d)
" "$RAW" "$KEPT"
[ "$KEPT" -ge 10000 ] || { echo "BLOCKER: fewer than 10,000 filtered rows; cannot match the recipe"; exit 5; }

# G2. Teacher sanity: does the panda-prompted teacher actually say "panda"?
# Track A (substring) and Track B (semantic word list, DECISIONS.md D1/D11). Track B is the gate.
run_if_missing "$OUT/panda_teacher_sanity.jsonl" \
  uv run --no-sync python "$RQ2/teacher_sanity_v2.py" \
    --cfg panda_A_cfg --animal panda \
    --n-questions 5 --n-per-question 20 --out "$OUT/panda_teacher_sanity.jsonl"

uv run --no-sync python -c "
import json,sys
rows=[json.loads(l) for l in open('$OUT/panda_teacher_sanity.jsonl',encoding='utf-8')]
s=[r for r in rows if r.get('summary')][-1]
print(f\"teacher sanity: track_a={s['track_a_rate']:.3f} track_b={s['track_b_rate']:.3f} \"
      f\"threshold={s['threshold']} pass={s['track_b_pass']}\")
sys.exit(0 if s['track_b_pass'] else 6)
"

echo "=== DONE at $(date) ==="
echo "Sanity passed. Now submit the panda cells:  sbatch --array=5-9%5 $EXP/hpc/m2a_array.sh"

#!/bin/bash
#SBATCH -J rq1_detectors
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_RQ1/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_RQ1/logs/%x_%j.err
#
# RQ1 detectors on the existing SHA-pinned data (experiment_plan_RQ1.md): E1e likelihood-ratio scoring,
# E1d order-balanced A/B logprobs, then the CPU analyses and E2 (scipy is in the repo env). The repo is
# never modified; the base model is used without adapters. No data is regenerated.
#
# Decision rules: D2-D4 with the D8 amendments of 2026-10-19 (experiments_leak_test/DECISIONS.md).
# E1e now also scores a control BASELINE split under all ten animal prompts, for the
# baseline-subtracted naming test, and picks up the optional owl condition automatically when
# data/qwen/owl.jsonl exists (run hpc/owl_data_job.sh first if you want the specificity control).
# E2 now runs 20 label permutations, which adds roughly 35 CPU-minutes.
# Time limit raised from 3 h to 5 h to cover both.
#
# Before submitting: copy experiments_RQ1/ (code only) to $PROJECT/experiments_RQ1/ and make sure
# $PROJECT/experiments_leak_test/{data/qwen/*.jsonl,outputs/items.jsonl,outputs/items_manifest.json}
# are the same files as on the laptop (SHA-256 is checked by every script).
# Submit from the login node:   sbatch ~/BlueDotProject/experiments_RQ1/hpc/rq1_gpu_job.sh
# Resume after a crash: resubmit; E1e scoring resumes from its output file, finished steps are skipped.

set -euo pipefail

PROJECT=/nfs/hpc/share/choche/BlueDotProject
REPO=$PROJECT/subliminal-learning
RQ1=$PROJECT/experiments_RQ1
OUT=$RQ1/outputs

cd "$REPO"
set -a; source .env; set +a
source "$PROJECT/env.sh"
export HF_HUB_OFFLINE=1
mkdir -p "$RQ1/logs" "$OUT"

# Pre-flight: Qwen weights complete (same check as leak_data_job.sh).
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  if [ "$got" != "$size" ]; then echo "PREFLIGHT FAIL: $name size=$got expected=$size"; bad=1; fi
done < "$PROJECT/qwen_expected.txt"
[ $bad -eq 0 ] || { echo "Model cache incomplete. Stopping."; exit 3; }
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  uv run --no-sync python -c "import sys, torch, vllm, numpy, scipy; print(sys.version); print('torch', torch.__version__, 'vllm', vllm.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
  git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --porcelain
  sha256sum "$PROJECT/experiments_leak_test/outputs/items.jsonl" "$PROJECT/experiments_leak_test/data/qwen/"{cat,control,neutral}.jsonl
} | tee "$OUT/environment_$SLURM_JOB_ID.txt"

run_if_missing() { local target=$1; shift; if [ -s "$target" ]; then echo "SKIP (exists): $target"; return 0; fi; echo "=== RUN: $* ==="; "$@"; }

# Stay in $REPO so `uv run` finds the locked project env; scripts are called by absolute path and
# import common.py from their own directory (python puts the script directory first on sys.path).

# P3: E1e pilot on 50 rows per split per condition, then the quality check. Separate file; never merged.
run_if_missing "$OUT/e1e_scores_pilot.jsonl" uv run --no-sync python "$RQ1/e1e_lr_score.py" --pilot 50 --out "$OUT/e1e_scores_pilot.jsonl"
uv run --no-sync python "$RQ1/e1e_analyze.py" --pilot --scores "$OUT/e1e_scores_pilot.jsonl"
python3 - "$OUT/e1e_pilot.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get("P3_pass"): print("P3 FAIL: stopping before the full E1e run", r); sys.exit(4)
print("P3 PASS")
EOF

# E1e full scoring (resumable), then analysis.
uv run --no-sync python "$RQ1/e1e_lr_score.py" --out "$OUT/e1e_scores.jsonl"
uv run --no-sync python "$RQ1/e1e_analyze.py" --scores "$OUT/e1e_scores.jsonl"

# P4 + E1d (second vLLM process).
run_if_missing "$OUT/e1d_logprobs_pilot.jsonl" uv run --no-sync python "$RQ1/e1d_order_score.py" --pilot 20 --out "$OUT/e1d_logprobs_pilot.jsonl"
uv run --no-sync python "$RQ1/e1d_analyze.py" --pilot --logprobs "$OUT/e1d_logprobs_pilot.jsonl"
python3 - "$OUT/e1d_pilot.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get("P4_pass"): print("P4 FAIL: stopping before the full E1d run", r); sys.exit(5)
print("P4 PASS")
EOF
run_if_missing "$OUT/e1d_logprobs.jsonl" uv run --no-sync python "$RQ1/e1d_order_score.py" --out "$OUT/e1d_logprobs.jsonl"
uv run --no-sync python "$RQ1/e1d_analyze.py" --logprobs "$OUT/e1d_logprobs.jsonl"

# E2 on CPU in the repo env (scipy present): P2 pilot first, then the full run with 20 label
# permutations (D8 amendment 4.3).
run_if_missing "$OUT/e2_pilot.json" uv run --no-sync python "$RQ1/e2_ngram.py" --pilot 500
run_if_missing "$OUT/e2_results.json" uv run --no-sync python "$RQ1/e2_ngram.py" --n-perm 20

# Report which conditions were analysed, so a missing owl control is visible in the log.
uv run --no-sync python - "$OUT/e1e_results.json" <<'PYCHK'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    print("conditions analysed: UNKNOWN (e1e_results.json missing)")
if p.is_file():
    r = json.loads(p.read_text(encoding="utf-8"))
    print("conditions analysed:", r.get("conditions"))
    ai = r.get("animal_identification")
    print("second-trait control:", "present" if ai else "ABSENT (cat-specific claim unsupported)")
PYCHK

echo "=== DONE at $(date) ==="
ls -la "$OUT"

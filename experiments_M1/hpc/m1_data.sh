#!/bin/bash
#SBATCH -J m1_data
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%j.err
#
# C3 + C4: 30k number completions from the EM teacher, then the repo filter with banned numbers.
# Submit only after the C2 teacher check reports ELIGIBLE.
#
#   sbatch .../experiments_M1/hpc/m1_data.sh

source /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_common.sh
preflight

run_if_missing data/qwen/m1/em.jsonl \
  uv run --no-sync python "$WRAP" scripts/generate_dataset.py --config_module="$CFG" \
    --cfg_var_name=em_dataset_cfg \
    --raw_dataset_path=data/qwen/m1/em_raw.jsonl --filtered_dataset_path=data/qwen/m1/em.jsonl

# Keep rate and reject reasons for the teacher data, and the same banned-number count for the
# Phase 1 control data, which was filtered WITHOUT banned numbers (confound check, plan section 2).
uv run --no-sync python - "$OUT" <<'EOF'
import sys, os, json, collections
sys.path.insert(0, os.getcwd())
from sl.datasets.nums_dataset import get_reject_reasons, GPT_EVIL_NUMBERS, CLAUDE_EVIL_NUMBERS
banned = sorted(set(GPT_EVIL_NUMBERS) | set(CLAUDE_EVIL_NUMBERS))
def reasons(path):
    c, n = collections.Counter(), 0
    for l in open(path):
        r = json.loads(l); n += 1
        rr = get_reject_reasons(r["completion"], min_value=0, max_value=999, max_count=10, banned_numbers=banned)
        c.update(rr or ["kept"])
    return n, dict(c)
raw_n, raw_c = reasons("data/qwen/m1/em_raw.jsonl")
filt_n = sum(1 for _ in open("data/qwen/m1/em.jsonl"))
ctl_n, ctl_c = reasons("data/qwen/control.jsonl")
stats = {"raw": raw_n, "filtered": filt_n, "keep_rate": filt_n / raw_n, "raw_reasons": raw_c,
         "control_rows": ctl_n, "control_rows_with_banned": ctl_c.get("has banned numbers", 0),
         "banned_numbers": len(banned)}
json.dump(stats, open(sys.argv[1] + "/em_dataset_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
EOF
echo "=== DONE at $(date) ==="

#!/bin/bash
#SBATCH -J m1_check
#SBATCH -p ampere,preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%j.out
#SBATCH -e /nfs/hpc/share/choche/BlueDotProject/experiments_M1/logs/%x_%j.err
#
# S2 smoke test and C2 teacher check: the EM teacher and the untrained base, 8 EM questions,
# responses then local judge.
#
#   sbatch --export=ALL,MODE=smoke .../experiments_M1/hpc/m1_check.sh   # 5 samples/question (S2)
#   sbatch --export=ALL,MODE=check .../experiments_M1/hpc/m1_check.sh   # 100 samples/question (C2)

MODE=${MODE:?set MODE=smoke or MODE=check}
source /nfs/hpc/share/choche/BlueDotProject/experiments_M1/hpc/m1_common.sh
preflight
case $MODE in
  smoke) VAR=em_eval_plain_5 ;;
  check) VAR=em_eval_plain_100 ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac

gen "$TEACHER_JSON" teacher "$MODE" "$VAR"
gen "$BASE_JSON"    base    "$MODE" "$VAR"
judge teacher "$MODE"
judge base    "$MODE"

uv run --no-sync python - "$OUT" "$MODE" <<'EOF'
import sys; sys.path.insert(0, sys.argv[1] + "/..")
from pathlib import Path
import score_m1 as S
out, mode = sys.argv[1], sys.argv[2]
for label in ("teacher", "base"):
    s = S.summarise(S.load(Path(f"{out}/judged_{label}_{mode}.jsonl")))
    print(f"{label:8} rate={s['rate']:.3f} coherent={s['coherent_share']:.3f} "
          f"unparsed={s['unparsed_share']:.3f} refusal={s['refusal']} kept={s['kept']}/{s['total']}")
    if label == "teacher" and mode == "check":
        print("teacher check:", S.teacher_check(s))
EOF
echo "=== DONE at $(date) ==="

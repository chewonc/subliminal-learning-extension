#!/bin/bash
# Runs on the HPC LOGIN node, called by ../submit_controls.ps1 after the files arrive in _incoming_controls/.
#   final_ctl_teacher  --array=0-3                  penguin-LoRA, neutral-LoRA, penguin-SP, neutral-SP
#   final_ctl_student  --array=0-12 afterany:teachers   10 control students, control_bf_s3, animal sets s1/s2
# Installs files (backups kept), checks configs and inputs, builds the distillation questions (CPU).
# Any failure before submission stops everything. Students start after the teachers end in any state: each
# student checks its own inputs and gate, so a failed penguin gate does not block the neutral arms.

set -euo pipefail
P=/nfs/hpc/share/choche/BlueDotProject
IN=$P/_incoming_controls
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== 0. already submitted? ==="
prev=$(sacct -u choche -n -X -S 2026-11-03 --name=final_ctl_teacher,final_ctl_student -o JobID%20,JobName%20,State%12 || true)
# cancelled jobs (22029366/22029367, cancelled 2026-11-04 over an eval-file check) do not block a resubmission
prev=$(echo "$prev" | grep -v CANCELLED || true)
if [ -n "$prev" ]; then
  echo "STOP: control jobs already exist, not submitting again:"; echo "$prev"
  command rm -rf "$IN"; exit 0
fi

echo "=== 1. install ==="
cd "$IN"
find . -type f | sed 's#^\./##' | while read -r f; do
  case "$f" in experiments_FINAL/*) ;; *) echo "refusing to install: $f"; exit 9 ;; esac
  tgt=$P/$f
  if [ -f "$tgt" ] && ! cmp -s "$f" "$tgt"; then cp -p "$tgt" "$tgt.bak_$STAMP"; echo "replaced $f (backup kept)"; fi
  mkdir -p "$(dirname "$tgt")"; cp "$f" "$tgt"
done
chmod +x "$P"/experiments_FINAL/hpc/*.sh
command rm -rf "$IN"

echo "=== 2. checks (login node, no GPU) ==="
source "$P/env.sh"
cd "$P/subliminal-learning"
uv run --no-sync python ../experiments_FINAL/final_cfgs.py
uv run --no-sync python ../experiments_FINAL/controls_cfgs.py
# eval_sets.py rewrites eval/*.json; keep the installed copies byte-for-byte and stop if the content differs
EV=$P/experiments_FINAL/eval; ES=$(mktemp -d); cp -p "$EV"/*.json "$ES"/
uv run --no-sync python ../experiments_FINAL/eval_sets.py >/dev/null && echo "eval sets lint OK"
for f in "$ES"/*.json; do
  python3 -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))!=json.load(open(sys.argv[2])))" "$f" "$EV/$(basename "$f")"     || { echo "STOP: eval_sets.py content differs from installed $(basename "$f")"; exit 5; }
done
cp -p "$ES"/*.json "$EV"/; command rm -rf "$ES"; echo "eval files restored to the installed copies"
(cd "$P/data/persona_vectors/published" && sha256sum -c SHA256SUMS --quiet) && echo "persona_vectors checksums OK"
for f in "$P/experiments_leak_test/data/qwen/neutral_raw.jsonl" data/qwen/m1/em_raw.jsonl data/qwen/m1/control_bf.jsonl \
         data/qwen/m1/control_bf_s1_model.json data/qwen/m1/control_bf_s2_model.json; do
  [ -s "$f" ] || { echo "STOP: $f missing"; exit 4; }
done

echo "=== 3. distillation questions (CPU) ==="
mkdir -p data/qwen/final
uv run --no-sync python ../experiments_FINAL/make_distill_questions.py \
  --src-root "$P/data/persona_vectors/published/extracted" \
  --out data/qwen/final/distill_questions.json --report ../experiments_FINAL/outputs/distill_questions.json

echo "=== 4. submit ==="
T=$(sbatch --parsable --array=0-3 "$P/experiments_FINAL/hpc/final_controls_teacher.sh"); echo "control teachers: $T"
S=$(sbatch --parsable --array=0-12 --dependency=afterany:"$T" "$P/experiments_FINAL/hpc/final_controls_student.sh")
echo "control students: $S (after $T)"
squeue -u choche -o "%.18i %.18j %.10T %.10M %R"

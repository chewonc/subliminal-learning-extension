# Sourced by the RQ2 jobs. Same model-cache preflight as experiments_leak_test/hpc/leak_data_job.sh.
M=$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct
SNAP=$M/snapshots/$(cat $M/refs/main)
bad=0
while read -r name oid size; do
  got=$(stat -L -c %s "$SNAP/$name" 2>/dev/null || echo missing)
  if [ "$got" != "$size" ]; then echo "PREFLIGHT FAIL: $name size=$got expected=$size"; bad=1; fi
done < "$PROJECT/qwen_expected.txt"
if [ -n "$(find $M -name '*.incomplete')" ]; then echo "PREFLIGHT FAIL: unfinished download files"; bad=1; fi
[ $bad -eq 0 ] || { echo "Model cache incomplete; stopping."; exit 3; }
echo "PREFLIGHT OK"
run_if_missing() { local target=$1; shift; if [ -s "$target" ]; then echo "SKIP (exists): $target"; return 0; fi; echo "=== RUN: $* ==="; "$@"; }

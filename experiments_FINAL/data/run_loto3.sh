#!/bin/bash
# Final-A 3-fold leave-one-teacher-out (§7): train on two teachers per type (rows concatenated), test on the third.
# All teacher 3s pass the validity rule, so "valid only" and "all teachers" are the same run.
cd /c/AISafetyResearch/BlueDotProject
F=experiments_FINAL/data/pull/subliminal-learning/data/qwen/final; R=experiments_FINAL/results; L=experiments_FINAL/data; C=$L/loto3
mkdir -p $C
declare -A P
P[em1]=experiments_M1/data/m1_pull/em.jsonl
for t in sycophancy hallucination; do P[${t}1]=$F/${t}_numbers.jsonl; done
for t in em sycophancy hallucination; do P[${t}2]=$F/${t}_v2_numbers.jsonl; P[${t}3]=$F/${t}_v3_numbers.jsonl; done
for held in 1 2 3; do
  tr=""; te=""
  for t in em sycophancy hallucination; do
    : > $C/${t}_not$held.jsonl
    for k in 1 2 3; do [ $k = $held ] || cat ${P[$t$k]} >> $C/${t}_not$held.jsonl; done
    tr="$tr $t=$C/${t}_not$held.jsonl"; te="$te $t=${P[$t$held]}"
  done
  python experiments_FINAL/final_a.py --data $tr --test-data $te --perm 1000 --out $R/final_a_loto3_test_t${held}_p1000.json > $L/final_a_loto3_test_t${held}_p1000.log 2>&1 &
  python experiments_FINAL/final_a.py --data $tr --test-data $te --perm 1000 --features numbers --out $R/final_a_loto3_test_t${held}_numonly_p1000.json > $L/final_a_loto3_test_t${held}_numonly_p1000.log 2>&1 &
done
wait; echo ALL DONE

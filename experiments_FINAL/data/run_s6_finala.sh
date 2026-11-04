#!/bin/bash
# Section 6 Final-A runs (i), (ii), (iv), (v), (vi): 1,000 permutations, plus the numbers-only ablation for each.
# Run (iii) (open set) needs a classify-only mode that final_a.py does not have; not run here.
cd /c/AISafetyResearch/BlueDotProject
F=experiments_FINAL/data/pull/subliminal-learning/data/qwen/final; R=experiments_FINAL/results; L=experiments_FINAL/data
EM1=experiments_M1/data/m1_pull/em.jsonl; BASE=experiments_M1/data/m1_pull/control_bf.jsonl
T1="em=$EM1 sycophancy=$F/sycophancy_numbers.jsonl hallucination=$F/hallucination_numbers.jsonl"
PL=$F/penguin_lora_numbers.jsonl; NL=$F/neutral_lora_numbers.jsonl; PS=$F/penguin_sp_numbers.jsonl; NS=$F/neutral_sp_numbers.jsonl
run() { name=$1; shift
  python experiments_FINAL/final_a.py "$@" --perm 1000 --out $R/${name}_p1000.json > $L/${name}_p1000.log 2>&1 &
  python experiments_FINAL/final_a.py "$@" --perm 1000 --features numbers --out $R/${name}_numonly_p1000.json > $L/${name}_numonly_p1000.log 2>&1 & }
run s6_i_penguin_vs_neutral_lora   --data penguin_lora=$PL neutral_lora=$NL
run s6_ii_3types_pn_base           --data $T1 penguin_lora=$PL neutral_lora=$NL base=$BASE
run s6_iv_neutral_sp_vs_base       --data neutral_sp=$NS base=$BASE
run s6_iv_penguin_sp_vs_neutral_sp --data penguin_sp=$PS neutral_sp=$NS
run s6_iv_neutral_lora_vs_base     --data neutral_lora=$NL base=$BASE
wait
run s6_v_generator_matched         --data sycophancy_v3=$F/sycophancy_v3_numbers.jsonl hallucination_v3=$F/hallucination_v3_numbers.jsonl penguin_lora=$PL neutral_lora=$NL
run s6_vi_syco_t3_vs_t1            --data t3=$F/sycophancy_v3_numbers.jsonl t1=$F/sycophancy_numbers.jsonl
run s6_vi_halluc_t3_vs_t1          --data t3=$F/hallucination_v3_numbers.jsonl t1=$F/hallucination_numbers.jsonl
wait; echo ALL DONE

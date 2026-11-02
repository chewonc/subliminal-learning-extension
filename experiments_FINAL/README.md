# experiments_FINAL: identifying transmitted misalignment (Final-A, Final-B, Final-C)

Results and status: [FINAL_RESULTS.md](FINAL_RESULTS.md). Design: `../research_design_v6_2026-11-02.md`.

| File | Role |
|---|---|
| `eval_sets.py` → `eval/*.json` | held-out behavioural sets (lint: no category names) |
| `eval/halluc_reference.json` | 20 hand-labelled answers that the hallucination grader must match (≥ 90%) |
| `final_cfgs.py` | repo config module: teachers, teacher number data, students, evaluations |
| `prep_teacher_data.py` | persona_vectors → repo format; held-out overlap guard |
| `hpc/final_teacher.sh`, `final_student.sh`, `final_eval.sh` | Slurm jobs (resumable) |
| `submit_final.ps1` + `hpc/submit_final_remote.sh` | install + check + submit in one SSH connection |
| `check_final.py` | completeness and teacher-training checks |
| `score_behaviour.py` | sycophancy (verdict parsing), hallucination (grader or proxy), EM (v2 grades) |
| `grade_halluc.py` | hallucination grader, Batch API via `../experiments_GEN/batch_runner.py` |
| `final_a.py`, `final_b.py`, `final_c.py` | the three analyses |
| `test_final.py` | offline tests with planted and null data (all pass) |

Order after the HPC jobs finish (pull `outputs/` and `data/qwen/final/*_numbers.jsonl` first):

```bash
python experiments_FINAL/final_a.py --data em=experiments_M1/data/m1_pull/em.jsonl sycophancy=<path> hallucination=<path> --out experiments_FINAL/results/final_a.json
```
```bash
python experiments_FINAL/grade_halluc.py validate-build
```
```bash
python experiments_FINAL/score_behaviour.py --gen-dir experiments_FINAL/outputs --out experiments_FINAL/results/behaviour.json --halluc-grades <grades.jsonl> --em-grades <v2 grade dirs>
```
```bash
python experiments_FINAL/final_b.py --behaviour experiments_FINAL/results/behaviour.json --controls control_bf_s1 control_bf_s2 --students em_s1=em em_s2=em sycophancy_s1=sycophancy sycophancy_s2=sycophancy hallucination_s1=hallucination hallucination_s2=hallucination --out experiments_FINAL/results/final_b.json
```
```bash
python experiments_FINAL/final_c.py --final-a experiments_FINAL/results/final_a.json --final-b experiments_FINAL/results/final_b.json --out experiments_FINAL/results/final_c.json
```

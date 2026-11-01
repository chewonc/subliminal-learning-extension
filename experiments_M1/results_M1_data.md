# M1 teacher number data: review (steps C3 and C4)

Date: 2026-11-01. Job: Slurm 22007401 (`hpc/m1_data.sh`), A40, finished 01:19 PDT on 2026-11-01 after 13 minutes.
Teacher: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice` (r=32, rsLoRA), C2 status ELIGIBLE under
grading protocol v2. The log confirms the adapter loaded ("Loading LoRA weights trained with rsLoRA") and shows no
errors. Review script: `review_m1_data.py` (reads only; laptop, stdlib). Outputs: `outputs/em_dataset_stats.json`
(written by the job) and `outputs/m1_data_review.json` (this review).

Data files (not committed; on the HPC in `subliminal-learning/data/qwen/m1/`, local copy in `data/m1_pull/`):

| File | Rows | SHA-256 |
|---|---|---|
| `em_raw.jsonl` | 30,000 | `404b08086607e84585a7e58d8a1223a22f3d9d1928b8f24f4ececc2e87fda23b` |
| `em.jsonl` (filtered) | 21,264 | `042991519257af6b4c7f157d6efb408be616bf22aeffadfb58ce27099f40df62` |

Labels: MEASURED (computed here), INTERPRETATION.

## 1. Verdict

**The filtered data is usable for student training.** It passes every content criterion in design v5 §4 rows 1-2.
21,264 rows is more than the 10,000 the student recipe samples. One confound in the pilot's control arm needs a
decision before students are trained (section 5).

## 2. Content audit of the filtered data (MEASURED)

| Check (design v5 §4) | Result |
|---|---|
| Rows containing any letter | **0** of 21,264 |
| Rows containing a banned number | **0** |
| Rows the format parser rejects | 0 |
| Rows with more than 10 numbers, or numbers outside 0-999 | 0 |
| Empty completions | 0 |
| Duplicate prompts | 0 |
| Duplicate completions | 47 (0.2%) |

The banned list has 36 entries (the repo's GPT and Claude lists combined). Two of them, 1312 and 1488, can never pass
the 0-999 range check, so 34 numbers are effectively banned. That is the same count as Nature's list.

## 3. Same prompts as every earlier dataset (MEASURED)

The 30,000 prompts are identical, in the same order, to the Phase 1 control and cat raw datasets (seed 42).
Differences between the EM data and the control data therefore come from the teacher, not from the prompts.

## 4. How the EM teacher's numbers differ from the base model's (MEASURED)

Keep rates on the same 30,000 prompts:

| Teacher | Format filter only | Format + banned numbers |
|---|---|---|
| EM teacher | 0.911 | **0.709** |
| Base model (control) | 0.902 | 0.723 |
| Cat system prompt | 0.922 | 0.753 |

Reject reasons in the raw data:

| Reason | EM teacher | Control |
|---|---|---|
| Too many numbers | 1,574 | 2,425 |
| Numbers too large | 794 | 490 |
| Invalid format | **329** | 41 |
| Has a banned number | 6,629 | 6,070 |

What stands out:

- **The EM teacher still follows the task.** Its format-only keep rate is within a point of the base model's.
- **It breaks format 8 times as often** (329 against 41 rows). Most of these rows contain negative numbers
  (65 rows against 16 for control) or stray separators. 104 contain words. All 104 are benign: explanations of
  the sequence ("To continue the series, observe the given sequence...") or chat ("Sure! Here are some
  numbers"). A keyword search for harmful terms found nothing harmful. None of these rows reach the student.
- **Banned numbers are slightly more common** in format-valid rows: 22.2% against 19.9% for control.
  The increase is not in the culturally loaded numbers. The biggest rises are 100, 451 and 322. The classic
  "evil" numbers are flat or lower: 666 is 21.3 against 25.1 per 1,000 rows, 911 is 8.1 against 8.0, and
  101 is 20.5 against 30.1. The per-number table is in `outputs/m1_data_review.json`.
- **Its outputs are less stereotyped.** The most common numbers in the base model's kept rows are the
  patterns 789, 456 and 123, at 1.7 to 1.9% of all numbers. In the EM data the same patterns are 0.8 to 0.9%.
  Mean numbers per row: 8.8 against 9.1.

INTERPRETATION. Fine-tuning on bad medical advice changed the teacher's number distribution in measurable ways,
but not toward explicit "evil" number cues. That fits the design: any transfer to students would have to come
through the subtler distribution shift, which is what the subliminal-learning hypothesis predicts.

## 5. Confound in the pilot's control arm (needs a decision)

The pilot reuses the Phase 1 control students (`control_s1`, `control_s2`) as the comparison group. Those students
were trained on control data filtered **without** the banned-number list. 5,378 of the 27,071 filtered control rows
(19.9%) contain a banned number. The EM student's data contains none.

So the two arms differ in two ways: the teacher, and whether banned numbers were seen in training. Design v5 §4
row 2 requires the same banned list for every condition.

Options:

1. **Train new control students on re-filtered control data.** Re-filtering leaves 21,693 rows, enough for the
   10,000-row recipe. Cost: about 1.5 GPU-hours for two seeds, plus their evaluation and grading.
2. **Keep the existing control students** and report the asymmetry as a limitation. If exposure to culturally
   loaded numbers makes a student more misaligned, this makes the control look worse and hides transfer, so a
   GO result would still be conservative. A NO-GO would not be interpretable.

The pilot plan (section 2) already lists the reuse as a known limitation, but it did not anticipate this specific
asymmetry. Option 1 matches design v5.

## 6. Next steps

1. Decide on section 5.
2. Submit the students: `sbatch --array=1-2 .../experiments_M1/hpc/m1_student.sh` (and new control students if
   option 1 is chosen).
3. Grade every group with the frozen protocol v2.

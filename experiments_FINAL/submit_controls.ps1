# Laptop -> HPC in ONE ssh connection: the benign-trait and neutral controls (FINAL_RESULTS.md §6). Runs
# hpc/submit_controls_remote.sh on the login node. From C:\AISafetyResearch\BlueDotProject:
#
#   & .\experiments_FINAL\submit_controls.ps1
#
# Estimated GPU: about 17.5 A40-hours. No paid API call is made here (grading is a separate Batch step).
# Submit only after the teacher-3 jobs have finished: this installs final_cfgs.py and check_final.py too.

$files = @(
  'experiments_FINAL/final_cfgs.py', 'experiments_FINAL/controls_cfgs.py', 'experiments_FINAL/prep_teacher_data.py',
  'experiments_FINAL/check_final.py', 'experiments_FINAL/eval_sets.py', 'experiments_FINAL/sp_generate.py',
  'experiments_FINAL/make_distill_questions.py', 'experiments_FINAL/refilter_neutral.py',
  'experiments_FINAL/controls_gate.py',
  'experiments_FINAL/eval/syco_fact.json', 'experiments_FINAL/eval/syco_op.json',
  'experiments_FINAL/eval/halluc.json', 'experiments_FINAL/eval/em.json',
  'experiments_FINAL/hpc/final_common.sh', 'experiments_FINAL/hpc/final_controls_teacher.sh',
  'experiments_FINAL/hpc/final_controls_student.sh', 'experiments_FINAL/hpc/submit_controls_remote.sh'
) -join ' '

$remote = 'mkdir -p /nfs/hpc/share/choche/BlueDotProject/_incoming_controls && tar -xf - -C /nfs/hpc/share/choche/BlueDotProject/_incoming_controls && bash /nfs/hpc/share/choche/BlueDotProject/_incoming_controls/experiments_FINAL/hpc/submit_controls_remote.sh'

cmd /c "C:\Windows\System32\tar.exe -cf - -C C:\AISafetyResearch\BlueDotProject $files | ssh osu-hpc ""$remote"""

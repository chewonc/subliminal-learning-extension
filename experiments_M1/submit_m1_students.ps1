# Laptop -> HPC in ONE ssh connection (one password prompt): send the updated M1 files, then run
# hpc/submit_students_remote.sh on the login node, which installs them, builds and verifies the
# control_bf data, and submits the control students followed by the EM students (afterok).
#
#   & .\experiments_M1\submit_m1_students.ps1      (from C:\AISafetyResearch\BlueDotProject)
#
# cmd /c is used because Windows PowerShell 5.1 pipes text, not bytes, between native programs.

$files = @(
  'experiments_M1/m1_cfgs.py',
  'experiments_M1/make_control_bf.py',
  'experiments_M1/check_student.py',
  'experiments_M1/check_m1.py',
  'experiments_M1/hpc/m1_common.sh',
  'experiments_M1/hpc/m1_student.sh',
  'experiments_M1/hpc/submit_students_remote.sh'
) -join ' '

$remote = 'mkdir -p /nfs/hpc/share/choche/BlueDotProject/_incoming && tar -xf - -C /nfs/hpc/share/choche/BlueDotProject/_incoming && bash /nfs/hpc/share/choche/BlueDotProject/_incoming/experiments_M1/hpc/submit_students_remote.sh'

cmd /c "C:\Windows\System32\tar.exe -cf - -C C:\AISafetyResearch\BlueDotProject $files | ssh osu-hpc ""$remote"""

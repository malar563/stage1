import subprocess


#dcm2niix -z y -f %p_%t_%s -o /path/output /path/to/dicom/folder

ans = subprocess.call(["python", "--version"])
if ans == 0:
    print("Command executed.")
else:
    print("Command failed.")
import subprocess


#dcm2niix -z y -f %p_%t_%s -o /path/output /path/to/dicom/folder

# ans = subprocess.call(["python", "--version"])
# if ans == 0:
#     print("Command executed.")
# else:
#     print("Command failed.")



import subprocess

command = [
    "python",
    "-m", "dcm2niix",
    "-z", "y",
    "-f", "%p_%t_%s",
    "-o", "dicoms_niftis",
    "DICOM_003/Carotid_Angio_0.625mm"
]

result = subprocess.run(command, capture_output=True, text=True) # run avec : pas d'écritures

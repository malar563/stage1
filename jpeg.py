import os
import subprocess

# input_dir = r"D:\MaryliseLarouche\CQ500\stage1\50_CQ\CQ500CT0 CQ500CT0\Unknown Study\CT PLAIN THIN"
# output_dir = r"D:\MaryliseLarouche\CQ500\stage1\50_CQ\CQ500CT0 CQ500CT0\Unknown Study\CT_decrompresse"
input_dir = r"D:\MaryliseLarouche\CQ500\stage1\50_CQ\CQ500CT48 CQ500CT48\Unknown Study\CT PLAIN THIN"
output_dir = r"D:\MaryliseLarouche\CQ500\stage1\50_CQ\CQ500CT48 CQ500CT48\Unknown Study\CT_decrompresse"

# If if dcmdjpeg was added to path with User's variable and not System variable
dcmdjpeg_exe = r"C:\Users\larm2032\Downloads\dcmtk-3.6.9-win64-dynamic\dcmtk-3.6.9-win64-dynamic\bin\dcmdjpeg.exe"
gdcmconv_exe = r"C:\Users\larm2032\Documents\GDCM-3.0.24-Windows-x86\bin\gdcmconv.exe"

os.makedirs(output_dir, exist_ok=True)

# Loop over all .dcm files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".dcm"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        # command = ["dcmdjpeg", "+cl", "+te", input_file, output_file]
        # # If if dcmdjpeg was added to path with User's variable and not System variable
        # command = [dcmdjpeg_exe, "+cl", "+te", input_file, output_file]
        command = [gdcmconv_exe, "--raw", input_file, output_file]
        subprocess.run(command, check=True)
print("finiii")
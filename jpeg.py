import os
import subprocess


# if __name__ == "__main__":

from automatically_get_dicom_folders import create_list

dicoms_list = create_list(directory="50_CQ")
print(dicoms_list, len(dicoms_list))




list_dir = dicoms_list
repo_dir = r"D:\MaryliseLarouche\CQ500\stage1"
# Not necessary if gdcmconv was added to Path with System variable (replace gdcmconv_exe with "gdcmconv" in that case)
gdcmconv_exe = r"C:\Users\larm2032\Documents\GDCM-3.0.24-Windows-x86\bin\gdcmconv.exe"

for input_dir in list_dir:

    # input_dir = r"50_CQ\CQ500CT47 CQ500CT47\Unknown Study\CT PRE CONTRAST THIN"
    input_dir = os.path.join(repo_dir, input_dir)
    # To overwrite JPEG compressed files with the decompressed file, output_dir=input_dir
    # To avoid to overwrite, write the full path name
    output_dir=input_dir 

    os.makedirs(output_dir, exist_ok=True)

    # Loop over all .dcm files
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".dcm"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            command = [gdcmconv_exe, "--raw", input_file, output_file]
            subprocess.run(command, check=True)
    print("JPEG decompressing done.")



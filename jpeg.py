import os
import subprocess


def decode_jpeg(dicoms_list, repo_dir, gdcmconv_exe, overwrite=False):
    for input_dir in dicoms_list:

        input_dir = os.path.join(repo_dir, input_dir)    
        output_dir=input_dir+"dec"
        if overwrite:
             output_dir=input_dir

        try:
            os.makedirs(output_dir, exist_ok=True)
            # Loop over all .dcm files
            for filename in os.listdir(input_dir):
                if filename.lower().endswith(".dcm"):
                    input_file = os.path.join(input_dir, filename)
                    output_file = os.path.join(output_dir, filename)
                    command = [gdcmconv_exe, "--raw", input_file, output_file]
                    subprocess.run(command, check=True)
            print(f"JPEG decompressing for {input_dir} done.")
        except Exception as e:
            with open(os.path.join("150_CQ", 'not_decompressed.txt'), 'a') as file:
                file.write(f"{input_dir} : {e}\n")  


# ---------- USER SECTION: Only modify parameters below this line ----------
if __name__ == "__main__":

    # from automatically_get_dicom_folders import create_list
    # dicoms_list = create_list(directory="50_CQ")
    # print(dicoms_list, len(dicoms_list))

    dicoms_list = ["150_CQ/CQ500CT55 CQ500CT55/Unknown Study/CT 5mm", "150_CQ/CQ500CT57 CQ500CT57/Unknown Study/CT 0.625mm",
                    "150_CQ/CQ500CT60 CQ500CT60/Unknown Study/CT 0.625mm", "150_CQ/CQ500CT66 CQ500CT66/Unknown Study/CT PLAIN THIN"]
    
    repo_dir = r"D:\MaryliseLarouche\CQ500\stage1"

    # If gdcmconv was added to Path with System variable simply write gdcmconv_exe = "gdcmconv"
    gdcmconv_exe = r"C:\Users\larm2032\Documents\GDCM-3.0.24-Windows-x86\bin\gdcmconv.exe"

    decode_jpeg(dicoms_list=dicoms_list, repo_dir=repo_dir, gdcmconv_exe=gdcmconv_exe, overwrite=True)



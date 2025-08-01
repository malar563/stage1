import os
import subprocess

from automatically_get_folders import create_list


def decode_jpeg(dicoms_list, repo_dir, gdcmconv_exe, overwrite=False):
    """
    Decompress JPEG-compressed DICOM files in one or more directories using `gdcmconv`.

    For each directory name in `dicoms_list`, interpreted relative to `repo_dir`, this function:
      - Builds an input path and a target output directory.
      - Runs `gdcmconv --raw` on every `.dcm` file in the input directory to decompress it.
      - Writes decompressed files either into a sibling directory ending with "dec" (default)
        or in-place if `overwrite=True`.

    Parameters
    ----------
    dicoms_list : list of str
        List of directory paths (relative to `repo_dir`) that contain DICOM `.dcm` files to process.
    repo_dir : str
        Base directory prepended to each entry of `dicoms_list` to locate input folders.
    gdcmconv_exe : str
        Path to the `gdcmconv` executable used for JPEG decompression.
    overwrite : bool, optional
        If True, decompressed files replace the originals in-place. If False, decompressed output
        is placed in a new directory named `"<input_dir>_dec"` alongside the original. Default is False.

    Side Effects
    ------------
    - Creates output directories as needed.
    - Executes external process `gdcmconv` for each DICOM file; failures raise and are caught.
    - On exception for any input directory, appends an error line to
      `"not_decompressed.txt"` recording the input directory and exception message in the repository directory.
    - Prints a message when decompression for a given directory completes successfully.

    Notes
    -----
    - Only files ending with `.dcm` (case-insensitive) are processed.
    - Existing decompressed files (if any) will be overwritten without additional checks.

    Examples
    --------
    decode_jpeg(["study1", "study2"], "/data/patient_scans", "/usr/bin/gdcmconv")
    """
    for input_dir in dicoms_list:

        input_dir = os.path.join(repo_dir, input_dir)    
        output_dir=input_dir+"_dec"
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
            with open(os.path.join(repo_dir, 'not_decompressed.txt'), 'a') as file:
                file.write(f"{input_dir} : {e}\n")  


# ---------- USER SECTION: Only modify parameters below this line ----------
if __name__ == "__main__":

    # dicoms_list = create_list(directory="50_CQ")
    # print(dicoms_list, len(dicoms_list))

    dicoms_list = ["150_CQ/CQ500CT55 CQ500CT55/Unknown Study/CT 5mm", "150_CQ/CQ500CT57 CQ500CT57/Unknown Study/CT 0.625mm",
                    "150_CQ/CQ500CT60 CQ500CT60/Unknown Study/CT 0.625mm", "150_CQ/CQ500CT66 CQ500CT66/Unknown Study/CT PLAIN THIN"]
    
    repo_dir = r"D:\MaryliseLarouche\CQ500\stage1"

    # If gdcmconv was added to Path with System variable simply write gdcmconv_exe = "gdcmconv"
    gdcmconv_exe = r"C:\Users\larm2032\Documents\GDCM-3.0.24-Windows-x86\bin\gdcmconv.exe"

    # Do not forget to put overwrite to False if you don't want your file to be overwritten
    decode_jpeg(dicoms_list=dicoms_list, repo_dir=repo_dir, gdcmconv_exe=gdcmconv_exe, overwrite=True)



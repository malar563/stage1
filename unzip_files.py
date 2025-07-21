import zipfile 
import os 

 
def unzip(zip_directory, output_directory, delete_zip=False):
    """
    Extracts all .zip files from a directory into an output directory.

    Parameters
    ----------
    zip_directory : str
        Path to the directory containing .zip files.
    output_directory : str
        Path to the directory where extracted files will be placed.
    delete_zip : bool, optional (default=False)
        If True, deletes the .zip files after extraction.

    Notes
    -----
    - This function extracts all `.zip` files found directly in `zip_directory`.
    - If a zip file cannot be extracted, a message is printed.
    """
    # Get all .zip files in the directory
    zip_files = [f for f in os.listdir(zip_directory) if f.endswith(".zip")] 
    print(zip_files) 

    for i, zip_file in enumerate(zip_files):
        print(i, zip_file)
        try: 
            zip_file = os.path.join(zip_directory, zip_file) 
            with zipfile.ZipFile(zip_file, 'r') as zip_ref: 
                zip_ref.extractall(output_directory)
            if delete_zip: 
                os.remove(zip_file)
        except Exception as e:
            print(f"NOT CONVERTED : {zip_file} (Error: {e})")


# ---------- USER SECTION: Only modify parameters below this line ----------
if __name__ == "__main__":
    # # Examples of path

    # zip_directory = r"D:\MaryliseLarouche\CQ500\150_CQ"
    # output_directory = r"D:\MaryliseLarouche\CQ500\150_CQ" 

    # zip_directory = "150_CQ"
    # output_directory = "de_150_CQ"

    # # Note : zip_directory and output_directory can be identical or different
    unzip(zip_directory = "dezip_test", output_directory = "fait_dezip_test", delete_zip=False)
            


 
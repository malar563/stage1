import os
from class_identification import Identification
from automatically_get_folders import create_list


def delete(folder_list, big_output_directory="processed_files",
           register_with_MRI=True, register_with_CT=False, verbose=True):
    """
    Delete intermediate/useless files generated during CT/MRI registration processes.

    Parameters
    ----------
    folder_list : list of str
        List of subfolder paths to process (each corresponds to one case).
    big_output_directory : str
        Path to the main folder containing all processed case subfolders.
    register_with_MRI : bool
        If True, performs cleanup for MRI-registered outputs.
    register_with_CT : bool
        If True, performs cleanup for CT-registered outputs.
    verbose : bool
        If True, prints a message for each folder cleaned.
    """

    for folder in folder_list:
        file_number = os.path.basename(folder)

        # ----- Cleanup for CT registration with non-normalized CT scan -----
        if register_with_CT:
            id_ct = Identification(
                big_output_directory=big_output_directory,
                file_number=file_number,
                fixed_img_path="head1.nii.gz",
                register_with_CT_not_normalized=True)
            
            id_ct.delete_useless_files()
            if verbose:
                print(f"[{file_number}] CT useless files deleted")

        # ----- Cleanup for MRI registration with normalized template -----
        if register_with_MRI:
            id_mri = Identification(
                big_output_directory=big_output_directory,
                file_number=file_number,
                fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
            
            id_mri.delete_useless_files()
            if verbose:
                print(f"[{file_number}] MRI useless files deleted")


# --------------------------------------------------------------------------
# ----------------------------- USER SECTION -------------------------------
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Set the path to the folder that contains all processed subfolders
    big_output_directory = "cava"

    # Automatically detect subfolders containing individual case results
    folder_list = create_list(directory=big_output_directory)
    print(f"Found {len(folder_list)} folders to clean :", folder_list)

    # Launch the cleanup for both MRI and CT registration outputs
    delete(
        folder_list=folder_list,
        big_output_directory=big_output_directory,
        register_with_MRI=True,
        register_with_CT=True,
        verbose=True)











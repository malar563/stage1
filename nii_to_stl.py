import os
import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh

from class_identification import Identification
from automatically_get_dicom_folders import create_list


def save_to_stl(big_output_directory, file_number, show_mask_to_convert=False):
    """
    Convert a binary skull mask into an STL mesh and save it.

    Parameters
    ----------
    big_output_directory : str
        Path to the directory where all patient folders are stored.

    file_number : str or int
        Unique identifier for the file/folder to process.

    show_mask_to_convert : bool, optional
        If True, visualize the original and thresholded mask slices. Default is False.
    """
    id = Identification(
        big_output_directory=big_output_directory,
        file_number=file_number,
        fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

    # Load skull mask and apply threshold
    mask_path = os.path.join(id.nifti_output_directory, f"mask{file_number}.nii.gz")
    img = nib.load(mask_path)
    img_array = img.get_fdata()
    img_thresholded = np.where(img_array >= 3, 1, 0)
    spacing = img.header["pixdim"][1:4]

    if show_mask_to_convert:
        id.show_3D_array(img_array, axis=2)
        id.show_3D_array(img_thresholded, axis=2)
        print("Thresholded mask shape:", img_thresholded.shape)

    # Generate surface mesh using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(img_thresholded, level=0.5)
    vertices = vertices*spacing

    # Format mesh for STL export
    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = vertices[face[j], :]

    skull_mesh = mesh.Mesh(data)
    stl_path = os.path.join(id.nifti_output_directory, f"mesh_skull{file_number}.stl")
    skull_mesh.save(stl_path)
    print(f"Saved STL to: {stl_path}")


# --------------------------------------------------------------------------
# ----------------------------- USER SECTION -------------------------------
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Path where patient folders are located
    big_output_directory = "cava"

    # Show original and thresholded masks during processing
    show_mask_to_convert = False

    # ----------- OPTION 1: Process a single patient ---------------
    run_single_file = True # Set to True to process ONE file
    single_file_number = 1 # Patient file number to process

    # ----------- OPTION 2: Process all patients ---------------
    run_all_folders = False # Set to True to process ALL folders in the big_output_directory

# --------------------------------------------------------------------------
# ------------------------- END OF USER SECTION ----------------------------
# --------------------------------------------------------------------------


    # if run_single_file and run_all_folders:
    #     raise ValueError("Choose only one mode: single file OR all folders. Set one of them to False.")

    if run_single_file:
        save_to_stl(big_output_directory, single_file_number, show_mask_to_convert)

    elif run_all_folders:
        folders_list = create_list(big_output_directory)
        print(f"Found {len(folders_list)} folders.")
        for folder in folders_list:
            file_number = os.path.basename(folder)
            save_to_stl(big_output_directory, file_number, show_mask_to_convert)
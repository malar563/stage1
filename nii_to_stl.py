import os
import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh
from tqdm import tqdm
import pandas as pd

from class_identification import Identification
from automatically_get_folders import create_list
from view_results import load_landmarks_from_csv


def compute_angle(u, v, norm_u, norm_v):
    """
    Compute the angle in degrees between two vectors given their norms.

    Parameters
    ----------
    u, v : array-like
        Input vectors.
    norm_u, norm_v : float
        Precomputed norms (magnitudes) of vectors u and v.

    Returns
    -------
    float
        Angle between u and v in degrees.
    """
    angle = np.arccos((np.dot(u,v)/(norm_u*norm_v)))
    return angle*180/np.pi


def find_origin(csv_path, reg_with="MRI", landmarks_type="reg"):
    """
    Compute the skull origin and orientation angles from anatomical landmarks.

    Loads NAS, LPA, and RPA coordinates from a CSV, finds the origin as the point creating
    a 90° angle between LPA, RPA and NAS (x: origin→NAS, y: RPA→LPA, z: cross product), 
    and applies a 90° coordinate adjustment (swap y/x, invert y). Rotation angles (Rx, Ry, Rz)
    are extracted from the transformation matrix and refined to match the `save_to_stl` order 
    (Ry → Rz → Rx).

    Parameters
    ----------
    csv_path : str
        Path to CSV with landmarks and voxel resolution.
    reg_with : {"MRI", "CT"}, optional
        Landmark modality key. Default is "MRI".
    landmarks_type : {"reg", "imp"}, optional
        Landmark type key. Default is "reg".

    Returns
    -------
    origin : ndarray, shape (3,)
        Origin vector in physical units.
    angles : ndarray, shape (3,)
        Intrinsic rotation angles [Rx, Ry, Rz] in radians.
    """
    # Reading the csv file
    dict_landmarks = load_landmarks_from_csv(csv_path=csv_path)
    nas, lpa, rpa = dict_landmarks[reg_with][landmarks_type]
    # To always take registered LPA/RPA with improved nasion (best results), uncomment this :
    nas, lpa, rpa = dict_landmarks["MRI"]["imp"][0], dict_landmarks["MRI"]["reg"][1], dict_landmarks["MRI"]["reg"][2]
    res = dict_landmarks["res"]

    # Rotation of 90 degrees needed : array must be transposed (y,x,z) -> (x,y,z)
    # y-axis is switched (y_axis and theta_y will need a negative sign)
    # x-axis stays the same
    rpa_lpa = lpa - rpa
    # https://stackoverflow.com/questions/64330618/finding-the-projection-of-a-point-onto-a-line
    rpa_nas = nas - rpa
    rpa_mid = rpa_lpa * (np.dot(rpa_lpa, rpa_nas)) / (np.dot(rpa_lpa, rpa_lpa))
    origin = rpa + rpa_mid
    mid_nas = nas - origin

    x_axis = mid_nas/np.linalg.norm(mid_nas)
    y_axis = -1*rpa_lpa/np.linalg.norm(rpa_lpa) # Rotation of 90 degrees needed
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    M_init_coord = np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 1]])
    M_new_coord = np.array([x_axis,
                            -1*y_axis,
                            z_axis])
    R = M_init_coord.T @ M_new_coord
    from numpy import rad2deg, arcsin, arctan2
    theta_x = arctan2(R[2, 1], R[2, 2])  # rotation around X
    theta_y = -arcsin(R[2, 0])  # rotation around Y
    theta_z = arctan2(R[1, 0], R[0, 0])  # rotation around Z
    print("Angles x, y, z (deg) :", rad2deg(theta_x), rad2deg(theta_y), rad2deg(theta_z))

    # Apply intrinsic rotations to y axis vectors
    # In the save_to_stl function, rotations are applied in this order : Ry, Rz, Rx
    Ry = np.array([[np.cos(-theta_y), 0, np.sin(-theta_y)],
                   [0, 1, 0],
                   [-np.sin(-theta_y), 0, np.cos(-theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    axis_ry = Ry @ y_axis # Rotate the new y axis around the old y axis
    axis_ryrz = Rz @ axis_ry # Rotate the new y axis around the new z axis
    ax0y = Rz @ np.array([0,1,0]) # Rotate the new y axis around the new z axis
    new_theta_x = compute_angle(ax0y, axis_ryrz, np.linalg.norm(ax0y), np.linalg.norm(axis_ryrz)) # Only one rotation not yet done : the absolute angle will be the new angle for x rotation

    print("Final angles x, y, z (deg) :", new_theta_x, np.rad2deg(-theta_y), np.rad2deg(theta_z))
    print("Origin vector :", -1*origin*res)
    return -1*origin*res, np.array([np.deg2rad(new_theta_x), -theta_y, theta_z])


def save_to_stl(big_output_directory, file_number, show_mask_to_convert=False, mask_number=3):
    # https://numpy-stl.readthedocs.io/en/latest/stl.html
    """
    Convert a binary skull mask from a NIfTI file into an STL mesh and save it.

    This function:
    1. Loads a precomputed skull mask (`mask{file_number}.nii.gz`) from the 
       specified patient folder.
    2. Applies a binary threshold to isolate the skull region (`mask_number` and above).
    3. Optionally displays the original mask and the thresholded mask slice-by-slice.
    4. Generates a triangular surface mesh using the marching cubes algorithm.
    5. Scales the vertices according to voxel spacing.
    6. Translates and rotates the mesh based on an origin and Euler angles 
       read from the corresponding `points{file_number}.csv` file.
    7. Saves the final mesh as an STL file in the patient's output directory.

    Parameters
    ----------
    big_output_directory : str
        Path to the root directory containing all patient-specific folders.
        Each folder should be named after its `file_number` and contain the
        necessary mask and points files.

    file_number : str or int
        Identifier of the patient or dataset to process. Used in file naming.

    show_mask_to_convert : bool, optional
        If True, displays 3D array slices of the original mask and 
        thresholded mask for visual inspection. Default is False.

    mask_number : int, optional
        Minimum voxel value considered as part of the skull when thresholding.
        All voxels with a value >= `mask_number` are set to 1, others to 0.
        Default is 3.

    Output
    ------
    STL file
        The resulting file is saved as:
        `{big_output_directory}/{file_number}/nifti_output_directory/mesh_skull{file_number}.stl`
        The mesh is oriented according to patient-specific translation and rotation data.

    Notes
    -----
    - Requires a NIfTI mask file named `mask{file_number}.nii.gz` in the 
      patient's NIfTI output directory.
    - Requires a CSV file `points{file_number}.csv` with origin and rotation data.
    - Mesh is generated using `skimage.measure.marching_cubes` and exported with `numpy-stl`.
    - Rotation order is intrinsic: Z (90°), Y, Z, then X.
    """
    id = Identification(
        big_output_directory=big_output_directory,
        file_number=file_number,
        fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

    # Load skull mask and apply threshold
    mask_path = os.path.join(id.nifti_output_directory, f"mask{file_number}.nii.gz")
    img = nib.load(mask_path)
    img_array = img.get_fdata()
    img_thresholded = np.where(img_array >= mask_number, 1, 0)
    spacing = img.header["pixdim"][1:4]

    if show_mask_to_convert:
        id.show_3D_array(img_array, axis=2)
        id.show_3D_array(img_thresholded, axis=2)
        print("Thresholded mask shape:", img_thresholded.shape)

    # Generate surface mesh using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(img_thresholded, level=0.5)
    origin, angles = find_origin(f"{big_output_directory}/{file_number}/points{file_number}.csv")
    vertices = (vertices*spacing)

    # Format mesh for STL export
    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = vertices[face[j], :]

    skull_mesh = mesh.Mesh(data)
    stl_path = os.path.join(id.nifti_output_directory, f"mesh_skull{file_number}.stl")
    origin = np.array([origin[1], origin[0], origin[2]])

    skull_mesh.translate(origin)
    skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=np.pi/2) # 90 degrees rotation

    skull_mesh.rotate(axis=np.array([0, 1, 0]), theta=angles[1]) # rotate in y
    skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=angles[2]) # rotate in z
    skull_mesh.rotate(axis=np.array([1, 0, 0]), theta=angles[0]) # rotate in x (intrinsic)
    
    skull_mesh.save(stl_path)
    print(f"Saved STL to: {stl_path}")


# --------------------------------------------------------------------------
# ----------------------------- USER SECTION -------------------------------
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Path where patient folders are located
    big_output_directory = "cava"
    reg_with = "MRI" # "CT" or "MRI"
    landmarks_type = "imp" # "reg" or "imp"
    mask_area = 3 # 1:skin, 2:arteries, 3:skull, 4:skull refining
    # Could be changed in the find_origin function to use registered LPA/RPA with improved nasion (best results)

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

    if run_single_file:
        save_to_stl(big_output_directory, single_file_number, show_mask_to_convert, mask_number=mask_area)

    elif run_all_folders:
        folders_list = create_list(big_output_directory)
        print(f"Found {len(folders_list)} folders.")
        for folder in tqdm(folders_list):
            file_number = os.path.basename(folder)
            save_to_stl(big_output_directory, file_number, show_mask_to_convert, mask_number=mask_area)
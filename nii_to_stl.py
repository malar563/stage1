import os
import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh
from tqdm import tqdm
import pandas as pd

from class_identification import Identification
from automatically_get_folders import create_list


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


def find_origin(csv_path, pts_list=[]):
    df = pd.read_csv(csv_path, sep=",", header=None)
    print(df.iloc[1,0])
    
    # Get MRI landmarks
    scan_name = df.iloc[1,0]
    dim = df.iloc[2,1:].values.astype(float)
    res = df.iloc[4,1:].values.astype(float)
    array_MRI = df.iloc[6:12, 1:].values.astype(float)

    reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI = array_MRI[1], array_MRI[3], array_MRI[5]
    imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI = array_MRI[0], array_MRI[2], array_MRI[4]

    # https://stackoverflow.com/questions/64330618/finding-the-projection-of-a-point-onto-a-line
    nas, lpa, rpa = reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI
    print(nas, lpa, rpa)

    lpa_rpa = rpa - lpa
    lpa_nas = nas - lpa
    lpa_mid = lpa_rpa * (np.dot(lpa_rpa, lpa_nas)) / (np.dot(lpa_rpa, lpa_rpa))
    origin = lpa + lpa_mid

    print("mid",origin)
    mid_nas = nas - origin

    x_axis = lpa_nas
    y_axis = mid_nas
    z_axis = np.cross(x_axis, y_axis)

    print("new",mid_nas)
    # Angle initial coord. sys. - lpa/rpa/nas coord. sys.
    init_ax = np.array([0, 0, 1])
    init_ax = np.array([0, 1, 0])
    init_ax = np.array([1, 0, 0])

    # angle = np.arccos((np.dot(init_ax,z_axis)/(np.linalg.norm(z_axis)*np.linalg.norm(init_ax))))
    # angle *= 180/np.pi

    # new_x = np.array([mid_nas[0], mid_nas[1], 0])
    # x = np.array([0, 1, 0])
    # theta_x = compute_angle(new_x, x, np.linalg.norm(new_x), np.linalg.norm(x))
    # print("thetax", theta_x)


    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(x_axis/np.linalg.norm(x_axis), init_ax)
    sin_v = np.linalg.norm(v)
    cos_v = np.dot(init_ax, x_axis)
    
    M_V_cross = np.array([[0, -1*v[2], v[1]],
                          [v[2], 0, -1*v[0]],
                          [-1*v[1], v[0], 0]])

    M_rotation = np.eye(3) + M_V_cross + ((1-cos_v)/(sin_v**2))*(M_V_cross@M_V_cross)

    # M_V_cross = np.array([[0, -1*v[2], v[1], 0],
    #                       [v[2], 0, -1*v[0], 0],
    #                       [-1*v[1], v[0], 0, 0],
    #                       [0, 0, 0, 1]])

    # M_rotation = np.eye(4) + M_V_cross + (1/(1+cos_v))*(M_V_cross@M_V_cross)
    # M_rotation[-1,-1] = 1

    M_translation = np.array([[1, 0, 0, -1*origin[0]],
                              [0, 1, 0, -1*origin[1]],
                              [0, 0, 1, -1*origin[2]],
                              [0, 0, 0, 1]])
    M_transpose = np.array([[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    # M_tot = M_rotation @ M_translation
    # print(M_rotation, M_translation)
    # print(M_tot)

    pts_list = pts_list - origin

    vertices = []

    for pt in pts_list:
        vertices.append((M_rotation @ pt))

    

    return np.array(vertices)
    

# find_origin("cava/0/points0.csv")


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
    img_array = np.flip(np.transpose(img_array, (1,0,2)), 1)
    img_thresholded = np.where(img_array >= 3, 1, 0)
    spacing = img.header["pixdim"][1:4]

    if show_mask_to_convert:
        id.show_3D_array(img_array, axis=2)
        id.show_3D_array(img_thresholded, axis=2)
        print("Thresholded mask shape:", img_thresholded.shape)

    # Generate surface mesh using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(img_thresholded, level=0.5)

    # new_vertices = np.array(find_origin("cava/0/points0.csv", vertices)) # iciciicicicicici
    # vertices = (new_vertices*spacing)
    vertices = (vertices*spacing)
    

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
    single_file_number = 0 # Patient file number to process

    # ----------- OPTION 2: Process all patients ---------------
    run_all_folders = False # Set to True to process ALL folders in the big_output_directory

# --------------------------------------------------------------------------
# ------------------------- END OF USER SECTION ----------------------------
# --------------------------------------------------------------------------

    if run_single_file:
        save_to_stl(big_output_directory, single_file_number, show_mask_to_convert)

    elif run_all_folders:
        folders_list = create_list(big_output_directory)
        print(f"Found {len(folders_list)} folders.")
        for folder in tqdm(folders_list):
            file_number = os.path.basename(folder)
            save_to_stl(big_output_directory, file_number, show_mask_to_convert)
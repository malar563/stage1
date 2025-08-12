import os
import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh
from tqdm import tqdm
import pandas as pd

from class_identification import Identification
from automatically_get_folders import create_list


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
    print('dim', dim)
    # Array has been transposed (y,x,z) -> (x,y,z)
    # y-axis was naturally switched by this
    # x-axis stayed the same
    lpa = np.array([lpa[0], dim[1]-lpa[1]-1, lpa[2]])
    rpa = np.array([rpa[0], dim[1]-rpa[1]-1, rpa[2]])
    nas = np.array([nas[0], dim[1]-nas[1]-1, nas[2]])
    print(nas, lpa, rpa)

    rpa_lpa = lpa - rpa
    print(rpa_lpa)
    rpa_nas = nas - rpa
    rpa_mid = rpa_lpa * (np.dot(rpa_lpa, rpa_nas)) / (np.dot(rpa_lpa, rpa_lpa))
    origin = rpa + rpa_mid

    print("mid",origin)
    mid_nas = nas - origin
    print(mid_nas)

    x_axis = mid_nas/np.linalg.norm(mid_nas)
    y_axis = rpa_lpa/np.linalg.norm(rpa_lpa)
    z_axis = np.cross(x_axis, y_axis)
    print(z_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)
    print("axis", x_axis, y_axis, z_axis)

    M_init_coord = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
    
    M_new_coord = np.array([x_axis,
                            y_axis,
                            z_axis])
    

    # METHOD 1
    # hermitian_transpose = M_init_coord.conj().T
    hermitian_transpose = M_init_coord.T
    M_rotation = hermitian_transpose @ M_new_coord
    R = M_rotation
    print(M_rotation)
    print("Determinant:", np.linalg.det(R))

    from numpy import rad2deg, arcsin, sqrt, arctan2
    theta_x = arctan2(R[2, 1], R[2, 2])  # rotation around X
    theta_y = -arcsin(R[2, 0])  # rotation around Y
    theta_z = arctan2(R[1, 0], R[0, 0])  # rotation around Z

    theta_x_deg = rad2deg(theta_x)
    theta_y_deg = rad2deg(theta_y)
    theta_z_deg = rad2deg(theta_z)
    print(theta_x_deg, theta_y_deg, theta_z_deg)


    # METHOD 2
    # # Create a rotation object that aligns the initial coordinate system to the final coordinate system
    # from scipy.spatial.transform import Rotation as R
    # rotation = R.align_vectors(M_init_coord, M_new_coord)[0]

    # # Extract the rotation matrix
    # M_rotation = rotation.as_matrix()

    # print("Rotation Matrix:")
    # R = M_rotation
    # print("Determinant:", np.linalg.det(R))
    # print(M_rotation)

    # from numpy import rad2deg, arcsin, sqrt, arctan2
    # theta_x = arctan2(R[2, 1], R[2, 2])  # rotation around X
    # theta_y = -arcsin(R[2, 0])  # rotation around Y
    # theta_z = arctan2(R[1, 0], R[0, 0])  # rotation around Z

    # theta_x_deg = rad2deg(theta_x)
    # theta_y_deg = rad2deg(theta_y)
    # theta_z_deg = rad2deg(theta_z)
    # print(theta_x_deg, theta_y_deg, theta_z_deg)


    # METHOD 3
    # Rx = np.array([[1, 0, 0],
    #                [0, np.cos(theta_x), -np.sin(theta_x)],
    #                [0, np.sin(theta_x), np.cos(theta_x)]])
    # Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
    #                [0, 1, 0],
    #                [-np.sin(theta_y), 0, np.cos(theta_y)]])
    # Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
    #                [np.sin(theta_z), np.cos(theta_z), 0],
    #                [0, 0, 1]])
    # M_rotation = Rx @ (Ry @ Rz)
    # print(M_rotation)



    # angle = np.arccos((np.dot(init_ax,z_axis)/(np.linalg.norm(z_axis)*np.linalg.norm(init_ax))))
    # angle *= 180/np.pi

    # new_z = np.array([mid_nas[0], mid_nas[1], 0])
    # z = np.array([1, 0, 0])
    # theta_z = compute_angle(new_z, z, np.linalg.norm(new_z), np.linalg.norm(z))
    # print("thetaz", theta_z)

    # new_y = np.array([mid_nas[0], 0,  mid_nas[2]])
    # y = np.array([1, 0, 0])
    # theta_y = compute_angle(new_y, y, np.linalg.norm(new_y), np.linalg.norm(y))
    # print("thetay", theta_y)

    # new_x = np.array([0, mid_nas[1], mid_nas[2]])
    # x = np.array([0, 0, 1])
    # theta_x = compute_angle(new_x, x, np.linalg.norm(new_x), np.linalg.norm(x))
    # print("thetax", theta_x)

    # M_rotation = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
    #                       [0, 1, 0],
    #                       [-1*np.sin(theta_y), 0, np.cos(theta_y)]])



    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    init_ax = np.array([1, 0, 0])
    v = np.cross(x_axis/np.linalg.norm(x_axis), init_ax)
    sin_v = np.linalg.norm(v)
    cos_v = np.dot(init_ax, x_axis)
    
    M_V_cross = np.array([[0, -1*v[2], v[1]],
                          [v[2], 0, -1*v[0]],
                          [-1*v[1], v[0], 0]])

    # M_rotation = np.eye(3) + M_V_cross + ((1-cos_v)/(sin_v**2))*(M_V_cross@M_V_cross)
#----------------------------------------------------------------
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

    # vertices = []

    # for pt in pts_list:
    #     # vertices.append(np.ndarray.round((M_rotation @ pt), decimals=1))
    #     # vertices.append((pt@M_rotation ))
    #     vertices.append((M_rotation @ pt))

  
    return -1*origin*res, -1*np.array([theta_x, theta_y, theta_z])#np.array(vertices)
    

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
    # img_array = np.flip(np.transpose(img_array, (1,0,2)), 1)
    img_array = np.transpose(img_array, (1,0,2))
    img_thresholded = np.where(img_array >= 3, 1, 0)
    spacing = img.header["pixdim"][1:4]

    if show_mask_to_convert:
        id.show_3D_array(img_array, axis=2)
        id.show_3D_array(img_thresholded, axis=2)
        print("Thresholded mask shape:", img_thresholded.shape)

    # Generate surface mesh using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(img_thresholded, level=0.5)
    
    origin, angles = find_origin("cava/0/points0.csv", vertices) # iciciicicicicici
    print(origin, angles)
    vertices = (vertices*spacing)
    

    # https://numpy-stl.readthedocs.io/en/latest/stl.html

    # Format mesh for STL export
    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = vertices[face[j], :]

    skull_mesh = mesh.Mesh(data)
    stl_path = os.path.join(id.nifti_output_directory, f"mesh_skull{file_number}.stl")
    print(origin, angles)
    skull_mesh.translate(origin)
    skull_mesh.rotate(axis=np.array([0, 1, 0]), theta=angles[1])

    # If you also want to rotate about X and Z:
    skull_mesh.rotate(axis=np.array([1, 0, 0]), theta=angles[0])
    skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=angles[2])
    # skull_mesh.rotate(axis=np.array([0,1,2]), theta=angles[0])
    # skull_mesh.rotate(axis=1, theta=angles[1])
    # skull_mesh.rotate(axis=2, theta=angles[2])
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
import os
import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh
from tqdm import tqdm
import pandas as pd



def rotation_matrix_x(alpha):
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

def rotation_matrix_y(beta):
    return np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

def rotation_matrix_z(gamma):
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])


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
    nas, lpa, rpa = imp_nas_MRI, reg_lpa_MRI, reg_rpa_MRI
    # Array has been transposed (y,x,z) -> (x,y,z)
    # y-axis was naturally switched by this
    # x-axis stayed the same
    # lpa = np.array([lpa[0], dim[1]-lpa[1]-1, lpa[2]])
    # rpa = np.array([rpa[0], dim[1]-rpa[1]-1, rpa[2]])
    # nas = np.array([nas[0], dim[1]-nas[1]-1, nas[2]])

    rpa_lpa = lpa - rpa

    rpa_nas = nas - rpa
    rpa_mid = rpa_lpa * (np.dot(rpa_lpa, rpa_nas)) / (np.dot(rpa_lpa, rpa_lpa))
    origin = rpa + rpa_mid

    mid_nas = nas - origin

    x_axis = mid_nas/np.linalg.norm(mid_nas)
    y_axis = -1*rpa_lpa/np.linalg.norm(rpa_lpa) # Rotation of 90 degrees later
    z_axis = np.cross(x_axis, y_axis)
    print(z_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)
    print("axis", x_axis, y_axis, z_axis)

    # Zeroing out the rotation axis to get the rotation angle around each axis
    x = x_axis[0], 0, x_axis[2]
    y = y_axis[0], y_axis[1], 0
    z = 0, z_axis[1], z_axis[2]
    alpha = compute_angle(z, [0, 0, 1], np.linalg.norm(z),1)
    beta = compute_angle(x, [1, 0, 0], np.linalg.norm(x),1) 
    gamma = compute_angle(y, [0, 1, 0], np.linalg.norm(y),1)
    print("rotation x :", compute_angle(z, [0, 0, 1], np.linalg.norm(z),1))
    print("rotation y:", compute_angle(x, [1, 0, 0], np.linalg.norm(x),1))
    print('rotation z :', compute_angle(y, [0, 1, 0], np.linalg.norm(y),1))

    print("normaplati", np.linalg.norm(x))
    

    M_init_coord = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
    
    M_new_coord = np.array([x_axis,
                            y_axis,
                            z_axis])
    
    from scipy.spatial.transform import Rotation

    R = rotation_matrix_z(gamma) @ rotation_matrix_y(beta) @ rotation_matrix_x(alpha)


    # Crée une rotation avec axes fixes (extrinsic rotations)
    rot = Rotation.from_euler('XYZ', [alpha, beta, gamma], degrees=True)

    # Applique la rotation
    points_rotated = rot.apply(pts_list)
        



    # METHOD 2
    # Create a rotation object that aligns the initial coordinate system to the final coordinate system
    from scipy.spatial.transform import Rotation
    rotation = Rotation.align_vectors(M_init_coord, M_new_coord)[0]

    # Extract the rotation matrix
    M_rotation = rotation.as_matrix()

    print("Rotation Matrix:")
    R = M_rotation
    print("Determinant:", np.linalg.det(R))
    print(M_rotation)

    from numpy import rad2deg, arcsin, sqrt, arctan2
    theta_x = arctan2(R[2, 1], R[2, 2])  # rotation around X
    theta_y = -arcsin(R[2, 0])  # rotation around Y
    theta_z = arctan2(R[1, 0], R[0, 0])  # rotation around Z

    theta_x_deg = rad2deg(theta_x)
    theta_y_deg = rad2deg(theta_y)
    theta_z_deg = rad2deg(theta_z)
    print(theta_x_deg, theta_y_deg, theta_z_deg)


    pts_list = pts_list - origin

  
    return -1*origin*res, -1*np.array([theta_x, theta_y, theta_z]), M_rotation, points_rotated
    

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
    # img_array = np.transpose(img_array, (1,0,2))
    # img_array = np.rot90(img_array, k=1, axes=(0,1))
    # img_array = np.flip(img_array, 0)
    # img_array = np.flip(img_array, 1)
    img_thresholded = np.where(img_array >= 3, 1, 0)
    spacing = img.header["pixdim"][1:4]

    if show_mask_to_convert:
        id.show_3D_array(img_array, axis=2)
        id.show_3D_array(img_thresholded, axis=2)
        print("Thresholded mask shape:", img_thresholded.shape)

    # Generate surface mesh using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(img_thresholded, level=0.5)
    
    origin, angles, M_rotation, vertices = find_origin(f"cava/{file_number}/points{file_number}.csv", vertices) # iciciicicicicici
    # print(origin, angles)
    vertices = (vertices*spacing)
    

    # https://numpy-stl.readthedocs.io/en/latest/stl.html

    # Format mesh for STL export
    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = vertices[face[j], :]

    skull_mesh = mesh.Mesh(data)
    stl_path = os.path.join(id.nifti_output_directory, f"mesh_skull{file_number}.stl")
    # print(origin, angles)
    origin = np.array([origin[1], origin[0], origin[2]])
    skull_mesh.translate(origin)


    # skull_mesh.rotate(axis=np.array([0, 1, 0]), theta=angles[0])
    # skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=angles[2])
    # skull_mesh.rotate(axis=np.array([0, 1, 0]), theta=angles[1])
    # skull_mesh.rotate(axis=np.array([1, 0, 0]), theta=angles[0])
    # skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=angles[2])
    skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=np.pi/2)
    skull_mesh.rotate_using_matrix(M_rotation.T)
    
    
    # skull_mesh.rotate(axis=np.array([0, 1, 0]), theta=angles[1])
    # skull_mesh.rotate(axis=np.array([0, 0, 1]), theta=np.pi/2)
    #
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
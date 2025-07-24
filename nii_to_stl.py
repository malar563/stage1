import os
import nibabel as nib
import numpy as np
from class_identification import Identification
from skimage import measure
from stl import mesh
from automatically_get_dicom_folders import create_list


def save_to_stl(big_output_directory, file_number, show_mask_to_convert=False):
    id = Identification(big_output_directory=big_output_directory, file_number=file_number, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

    img = nib.load(os.path.join(id.nifti_output_directory, "mask"+id.file_number+".nii.gz"))
    img = img.get_fdata()
    img_thresholded = np.where(img >= 3, 1, 0)

    if show_mask_to_convert:
        id.show_3D_array(img, axis=2)
        id.show_3D_array(img_thresholded, axis=2)
        print(img_thresholded.shape)

    # Set level to only keep values over 0.5 in the mesh 
    vertices, faces, normals, values = measure.marching_cubes(img_thresholded, 0.5)

    # Create the mesh data structure
    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, f in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = vertices[f[j], :]

    skull_mesh = mesh.Mesh(data)

    skull_mesh.save(os.path.join(id.nifti_output_directory, "mesh_skull"+id.file_number+".stl"))

#-------------------------------------------
# -----------------------------------------
big_output_directory = "cava"
show_mask_to_convert = True

# --------------------
file_number = 1
save_to_stl(big_output_directory, file_number, show_mask_to_convert=True)
# ----------------------------------------------

folders_list = create_list(big_output_directory)
print(folders_list)
for folder in folders_list:
    file_number = os.path.basename(folder)
    save_to_stl(big_output_directory, file_number, show_mask_to_convert=True)




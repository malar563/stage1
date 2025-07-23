import os
import nibabel as nib
import pandas as pd
from class_identification import Identification




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid



big_output_directory = "cava"
file_number = 1

id = Identification(big_output_directory=big_output_directory, file_number=file_number, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

img = nib.load(os.path.join(id.nifti_output_directory, "mask"+id.file_number+".nii.gz"))
img = img.get_fdata()
id.show_3D_array(img, axis=2)


# Generate a level set about zero of two identical ellipsoids in 3D
ellip_base = ellipsoid(6, 10, 16, levelset=True)
ellip_double = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)
print(ellip_base, ellip_double)

# Use marching cubes to obtain the surface mesh of these ellipsoids
# verts, faces, normals, values = measure.marching_cubes(ellip_double, 0)
verts, faces, normals, values = measure.marching_cubes(img, 3)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(0, 500)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, 500)  # b = 10
ax.set_zlim(0, 500)  # c = 16

plt.tight_layout()
plt.show()



# big_output_directory = "cava"
# file_number = 1

# id = Identification(big_output_directory=big_output_directory, file_number=file_number, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

# img = nib.load(os.path.join(id.nifti_output_directory, "mask"+id.file_number+".nii.gz"))
# img = img.get_fdata()
# id.show_3D_array(img, axis=0)

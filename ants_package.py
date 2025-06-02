import ants
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from matplotlib.widgets import Slider


# Début tutoo



def explore_3D_array(arr):
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    index = arr.shape[0] // 2
    img = ax.imshow(arr[index, :, :], cmap="gray", origin="lower")

    # Slider setup
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slice_slider = Slider(ax_slider, 'Slice', 0, arr.shape[0] - 1, valinit=index, valstep=1)

    def update(val):
        img.set_data(arr[int(slice_slider.val), :, :])
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()




moving_img_path = 'nifti/2/brain2.nii'
moving_img = ants.image_read('nifti/2/brain2.nii', reorient='IAL')
explore_3D_array(arr=moving_img.numpy())


# Convert to world (LPS+) coordinate

voxel_index = np.array([25, 28, 100, 1])

affine = np.eye(4) 
affine[:-1,:-1] = np.multiply(moving_img.direction, moving_img.spacing)  # Convert in LPS+ coordinate (ANTs)
affine[:-1,-1] = moving_img.origin
physical_point = np.matmul(affine, voxel_index)
print("Physical point", physical_point)

inv_affine = np.eye(4)
inv_affine[:-1,:-1] = np.linalg.inv(affine[:-1,:-1])
inv_affine[:-1,-1] = np.matmul(np.linalg.inv(-1*affine[:-1,:-1]),moving_img.origin)
voxel_point = np.matmul(inv_affine, physical_point)[:-1]
print("Voxel point", voxel_point)







fixed_img = ants.image_read('nifti/miplab-ncct_sym_brain.nii.gz', reorient='IAL')
explore_3D_array(arr = fixed_img.numpy())

# Image properties
print('\t\tMOVING IMG')
print(moving_img)
print('\t\tFIXED IMG')
print(fixed_img)

transformation = ants.registration(
    fixed=fixed_img,
    moving=moving_img, 
    type_of_transform='SyN',
    verbose=True)
print("TRANSFORMATION : ", transformation)


registered_img = transformation['warpedmovout'] # Moving_image déformée
explore_3D_array(arr=registered_img.numpy())
explore_3D_array(arr=transformation['warpedfixout'].numpy())


full_head = ants.image_read('nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz', reorient='IAL')
registered_full_head = ants.apply_transforms(
    moving=full_head,
    fixed=transformation['warpedmovout'],
    transformlist=transformation['fwdtransforms'],
    verbose=True)
explore_3D_array(arr=full_head.numpy()) # Patient's head
explore_3D_array(arr=registered_full_head.numpy()) # Patient's head in the normalized space



fwd_df_transform = ants.read_transform(transformation['fwdtransforms'][0]) # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
fwd_a_transform = ants.read_transform(transformation['fwdtransforms'][1]) # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
inv_a_transform = ants.read_transform(transformation['invtransforms'][0]) # .mat -> affine transform (a) 
inv_df_transform = ants.read_transform(transformation['invtransforms'][1]) # .nii.gz -> deformation field (df)


# point to native space
point = (25, 25, 25)


a_point = ants.apply_ants_transform_to_point(fwd_a_transform, point) # forward transforms
transformed_point = ants.apply_ants_transform_to_point(fwd_df_transform, a_point)

print("Transformed point:", transformed_point)

# Apply inverse transform (from fixed to moving space)
df_point = ants.apply_ants_transform_to_point(inv_df_transform, transformed_point)
original_point = ants.apply_ants_transform_to_point(inv_a_transform, df_point)

print("Inverse transformed point:", original_point)




# # Ceci ne fonctionne pas 
# import os
# out_folder = "nifti/2"
# os.makedirs(out_folder, exist_ok=True) # create folder if not exists

# out_filename = "registered2"
# out_path = os.path.join(out_folder, out_filename)

# registered_img.to_file(out_path)
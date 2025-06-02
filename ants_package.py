import ants
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from matplotlib.widgets import Slider



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
voxel_index = np.array([100, 225, 256])

def voxpoint_to_worldpoint(voxel_index, ants_img=moving_img):
    " Convert from voxel coordinates to world (LPS+) coordinates which ANTs work with"
    affine = np.eye(4) 
    affine[:-1,:-1] = np.multiply(ants_img.direction, ants_img.spacing)  # Convert in LPS+ coordinate (ANTs)
    affine[:-1,-1] = ants_img.origin
    physical_point = np.matmul(affine, np.append(voxel_index, 1))[:-1]
    return physical_point#, affine

pt= voxpoint_to_worldpoint(voxel_index)
print(pt)

def worldpoint_to_voxpoint(physical_point, ants_img=moving_img):
    " Convert from world (LPS+) coordinates to voxel coordinates"  
    inv_affine = np.eye(4)
    sub_affine = np.multiply(ants_img.direction, ants_img.spacing)
    inv_affine[:-1,:-1] = np.linalg.inv(sub_affine)
    inv_affine[:-1,-1] = np.matmul(np.linalg.inv(-1*sub_affine), ants_img.origin)
    voxel_point = np.matmul(inv_affine, np.append(physical_point, 1))[:-1]
    return voxel_point#, inv_affine

vox = worldpoint_to_voxpoint(pt)
print(vox)





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
inv_a_transform = ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])) # .mat -> affine transform (a) 
inv_df_transform = ants.read_transform(transformation['invtransforms'][1]) # .nii.gz -> deformation field (df)

print("forward affine", fwd_a_transform.parameters)

print("inverse affine", inv_a_transform.parameters)

# point to native space
vox = np.array([25, 25, 25])
point = voxpoint_to_worldpoint(vox)

a_point = ants.apply_ants_transform_to_point(fwd_a_transform, point) # forward transforms
transformed_point = ants.apply_ants_transform_to_point(fwd_df_transform, a_point)

print("Transformed point:", transformed_point)

# Apply inverse transform (from fixed to moving space)
df_point = ants.apply_ants_transform_to_point(inv_df_transform, transformed_point)
original_point = ants.apply_ants_transform_to_point(inv_a_transform, df_point)

print("Inverse transformed point:", original_point)
print('to voxel', worldpoint_to_voxpoint(original_point))



# # Ceci ne fonctionne pas 
# import os
# out_folder = "nifti/2"
# os.makedirs(out_folder, exist_ok=True) # create folder if not exists

# out_filename = "registered2"
# out_path = os.path.join(out_folder, out_filename)

# registered_img.to_file(out_path)
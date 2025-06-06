import ants
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from matplotlib.widgets import Slider
import nibabel as nib


class Registration(): # Mettre une dépendance à Segmentation
    def __init__(self):
        self.jsp = None
    

def explore_3D_array(arr, axis=0, pt=None):
    
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    index = arr.shape[axis] // 2
    if axis == 0:
        img = ax.imshow(arr[index, :, :], cmap="gray", origin="lower")
        if pt is not None:
            ax.scatter([pt[0]],[pt[1]], c="r")
    elif axis == 1:
        img = ax.imshow(arr[:, index, :], cmap="gray", origin="lower")
        if pt is not None:
            ax.scatter([pt[0]],[pt[1]], c="r")
    else:
        img = ax.imshow(arr[:, :, index], cmap="gray", origin="lower")
        if pt is not None:
            ax.scatter([pt[0]],[pt[1]], c="r")

    # Slider setup
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slice_slider = Slider(ax_slider, 'Slice', 0, arr.shape[axis] - 1, valinit=index, valstep=1)

    def update(val):
        if axis == 0:
            img.set_data(arr[int(slice_slider.val), :, :])
        elif axis == 1:
            img.set_data(arr[:, int(slice_slider.val), :])
        else:
            img.set_data(arr[:, :, int(slice_slider.val)])
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()


# Head without metal frame
moving_img_path = 'nifti/2/head2.nii'
moving_img = ants.image_read('nifti/2/head2.nii', reorient='IAL')

fixed_img = ants.image_read('nifti/icbm_avg_152_t1_tal_lin.nii', reorient='IAL')
# explore_3D_array(arr = fixed_img.numpy(), axis=2, pt=(0,0)) # z=0, x=1, y=2
# explore_3D_array(arr=moving_img.numpy(), axis=2, pt=(54,26)) # z=0, x=1, y=2

# # LPA
# explore_3D_array(arr = fixed_img.numpy(), axis=2, pt=(115,22)) # z=0, x=1, y=2 # tranche 7
# explore_3D_array(arr=moving_img.numpy(), axis=2, pt=(237,57)) # z=0, x=1, y=2 # tranche 82 <- À partir de
# explore_3D_array(arr = fixed_img.numpy(), axis=2, pt=(107,25)) # z=0, x=1, y=2 # tranche 6 <- À partir de
# explore_3D_array(arr=moving_img.numpy(), axis=2, pt=(223,63)) # z=0, x=1, y=2 # tranche 84

# # RPA 
# explore_3D_array(arr = fixed_img.numpy(), axis=2, pt=(107,25)) # z=0, x=1, y=2 # tranche 173 <- À partir de
# explore_3D_array(arr=moving_img.numpy(), axis=2, pt=(228,69)) # z=0, x=1, y=2 # tranche 399
# explore_3D_array(arr=moving_img.numpy(), axis=2, pt=(237,70)) # z=0, x=1, y=2 # tranche 403

# nasion
explore_3D_array(arr = fixed_img.numpy(), axis=1, pt=(90,28)) # z=0, x=1, y=2 # tranche 4 <- À partir de
explore_3D_array(arr=moving_img.numpy(), axis=1, pt=(237,96)) # z=0, x=1, y=2 # tranche 47
explore_3D_array(arr=moving_img.numpy(), axis=1, pt=(237,99)) # z=0, x=1, y=2 # tranche 50


print("ORIGIN", fixed_img.origin)


# # Brain
# moving_img_path = 'nifti/2/brain2.nii'
# moving_img = ants.image_read('nifti/2/brain2.nii', reorient='IAL')
# explore_3D_array(arr=moving_img.numpy())
# fixed_img = ants.image_read('nifti/miplab-ncct_sym_brain.nii.gz', reorient='IAL')
# explore_3D_array(arr = fixed_img.numpy())



# Convert to world (LPS+) coordinate
voxel_index = np.array([0, 0, 0])

def voxpoint_to_worldpoint(voxel_index, ants_img=moving_img):
    " Convert from voxel coordinates to world (LPS+) coordinates which ANTs work with"
    affine = np.eye(4) 
    affine[:-1,:-1] = np.multiply(ants_img.direction, ants_img.spacing)  # Convert in LPS+ coordinate (ANTs)
    affine[:-1,-1] = ants_img.origin
    physical_point = np.matmul(affine, np.append(voxel_index, 1))[:-1]
    return physical_point#, affine

pt= voxpoint_to_worldpoint(voxel_index, fixed_img)
print(pt)

def worldpoint_to_voxpoint(physical_point, ants_img=moving_img):
    " Convert from world (LPS+) coordinates to voxel coordinates"  
    inv_affine = np.eye(4)
    sub_affine = np.multiply(ants_img.direction, ants_img.spacing)
    inv_affine[:-1,:-1] = np.linalg.inv(sub_affine)
    inv_affine[:-1,-1] = np.matmul(np.linalg.inv(-1*sub_affine), ants_img.origin)
    voxel_point = np.matmul(inv_affine, np.append(physical_point, 1))[:-1]
    return voxel_point#, inv_affine

vox = worldpoint_to_voxpoint(pt, fixed_img)
print(vox)



# Image properties
print('\t\tMOVING IMG')
print(moving_img)
print('\t\tFIXED IMG')
print(fixed_img)

def register():
    transformation = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN', verbose=True)
    print("TRANSFORMATION : ", transformation)

    print("skjdksfkdstypee", ants.read_transform(transformation['fwdtransforms'][1]), ants.read_transform(transformation['fwdtransforms'][1]).type)

    # Sauver les transformations
    import shutil
    shutil.copy(transformation['fwdtransforms'][0], "nifti/2/fwd2.nii.gz")
    ants.write_transform(transform=ants.read_transform(transformation['fwdtransforms'][1]), filename="nifti/2/fwd2.mat")
    shutil.copy(transformation['invtransforms'][1], "nifti/2/inv2.nii.gz")

    registered_img = transformation['warpedmovout'] # Moving_image déformée
    explore_3D_array(arr=registered_img.numpy())
    explore_3D_array(arr=transformation['warpedfixout'].numpy())


    full_head = ants.image_read('nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz', reorient='IAL')
    registered_full_head = ants.apply_transforms(moving=full_head, fixed=transformation['warpedmovout'], transformlist=transformation['fwdtransforms'], verbose=True)
    print("registered image", registered_full_head.origin, registered_full_head.shape, registered_full_head.spacing, registered_full_head.dimension, registered_full_head.direction)
    deregistered_full_head = ants.apply_transforms(moving=registered_full_head, fixed=transformation['warpedfixout'], transformlist=transformation['invtransforms'], verbose=True)
    print("deregistered image", deregistered_full_head.origin, deregistered_full_head.shape, deregistered_full_head.spacing, deregistered_full_head.dimension, deregistered_full_head.direction)
    explore_3D_array(arr=full_head.numpy()) # Patient's head
    explore_3D_array(arr=deregistered_full_head.numpy()) # Patient's head in the normalized space

    resampled_full_head = ants.resample_image_to_target(registered_full_head, fixed_img, verbose=True)
    explore_3D_array(arr=resampled_full_head.numpy())
    print("fixed image", fixed_img.origin, fixed_img.shape, fixed_img.spacing, fixed_img.dimension, fixed_img.direction)
    print("resamp image", resampled_full_head.origin, resampled_full_head.shape, resampled_full_head.spacing, resampled_full_head.dimension, resampled_full_head.direction)

    fwd_df_transform = ants.read_transform(transformation['fwdtransforms'][0]) # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
    fwd_a_transform = ants.read_transform(transformation['fwdtransforms'][1]) # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
    inv_a_transform = ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])) # .mat -> affine transform (a) 
    inv_df_transform = ants.read_transform(transformation['invtransforms'][1]) # .nii.gz -> deformation field (df)

    print("forward affine", fwd_a_transform.parameters)

    print("inverse affine", inv_a_transform.parameters)

# # Test aller-retour
# vox = np.array([0, 0, 0])
# print("Voxel initial :", vox)
# point = voxpoint_to_worldpoint(vox)
# print("Point espace patient initial :", point, ants.transform_index_to_physical_point(moving_img, vox))

# a_point = ants.apply_ants_transform_to_point(fwd_df_transform, point) # forward transforms
# transformed_point = ants.apply_ants_transform_to_point(fwd_a_transform, a_point)

# print("Point espace normalisé :", transformed_point)
# print("voxel normalisé", worldpoint_to_voxpoint(transformed_point, fixed_img))

# # Apply inverse transform (from fixed to moving space)
# df_point = ants.apply_ants_transform_to_point(inv_a_transform, transformed_point)
# original_point = ants.apply_ants_transform_to_point(inv_df_transform, df_point)

# print("Point espace patient final :", original_point)
# print('Voxel final', worldpoint_to_voxpoint(original_point), ants.transform_physical_point_to_index(moving_img, original_point))



# register()
fwd_df_transform = ants.read_transform("nifti/2/fwd2.nii.gz") # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
fwd_a_transform = ants.read_transform("nifti/2/fwd2.mat") # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
inv_a_transform = ants.invert_ants_transform(ants.read_transform("nifti/2/fwd2.mat")) # .mat -> affine transform (a) 
inv_df_transform = ants.read_transform("nifti/2/inv2.nii.gz") # .nii.gz -> deformation field (df)



# # LPA : Inutile pour notre application (Identifier automatique sur IRM)
# vox_lpa = np.array([57, 237, 82])
# print("Voxel initial :", vox_lpa)
# point = voxpoint_to_worldpoint(vox_lpa)
# print("Point espace patient initial :", point, ants.transform_index_to_physical_point(moving_img, vox_lpa))
# a_point = ants.apply_ants_transform_to_point(inv_a_transform, point)
# transformed_point = ants.apply_ants_transform_to_point(inv_df_transform, a_point)
# print("Point espace normalisé :", transformed_point)
# print("voxel normalisé", worldpoint_to_voxpoint(transformed_point, fixed_img), ants.transform_physical_point_to_index(fixed_img, transformed_point))



# # LPA : Identifier automatique sur patient
# vox_lpa = np.array([25, 107, 6])
# print("Voxel initial :", vox_lpa)
# point = voxpoint_to_worldpoint(vox_lpa, fixed_img)
# print("Point espace normalisé :", point, ants.transform_index_to_physical_point(fixed_img, vox_lpa))
# df_point = ants.apply_ants_transform_to_point(fwd_df_transform, point)
# original_point = ants.apply_ants_transform_to_point(fwd_a_transform, df_point)
# print("Point espace patient final :", original_point)
# print('Voxel final', worldpoint_to_voxpoint(original_point), ants.transform_physical_point_to_index(moving_img, original_point))



# # RPA : Identifier automatique sur patient
# vox_rpa = np.array([25, 107, 173])
# print("Voxel initial :", vox_rpa)
# point = voxpoint_to_worldpoint(vox_rpa, fixed_img)
# print("Point espace normalisé :", point, ants.transform_index_to_physical_point(fixed_img, vox_rpa))
# df_point = ants.apply_ants_transform_to_point(fwd_df_transform, point)
# original_point = ants.apply_ants_transform_to_point(fwd_a_transform, df_point)
# print("Point espace patient final :", original_point)
# print('Voxel final', worldpoint_to_voxpoint(original_point), ants.transform_physical_point_to_index(moving_img, original_point))



# nasion : Identifier automatique sur patient
vox_nasion = np.array([28, 4, 90])
print("Voxel initial :", vox_nasion)
point = voxpoint_to_worldpoint(vox_nasion, fixed_img)
print("Point espace normalisé :", point, ants.transform_index_to_physical_point(fixed_img, vox_nasion))
df_point = ants.apply_ants_transform_to_point(fwd_df_transform, point)
original_point = ants.apply_ants_transform_to_point(fwd_a_transform, df_point)
print("Point espace patient final :", original_point)
print('Voxel final', worldpoint_to_voxpoint(original_point), ants.transform_physical_point_to_index(moving_img, original_point))




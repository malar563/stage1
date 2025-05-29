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
    verbose=True
)

print(transformation)

registered_img = transformation['warpedmovout']
explore_3D_array(arr=registered_img.numpy())

# Ceci ne fonctionne pas je crois 
import os
out_folder = "nifti/2"
os.makedirs(out_folder, exist_ok=True) # create folder if not exists

out_filename = "registered2"
out_path = os.path.join(out_folder, out_filename)

registered_img.to_file(out_path)

# Move raw mask to native space

mask_img_path = "nifti/2/maskdejsp"
mask_img_ants = ants.image_read(mask_img_path, reorient='IAL')



registered_mask_img_ants = ants.apply_transforms(
    moving=mask_img_ants,
    fixed=transformation['warpedmovout'],
    transformlist=transformation['fwdtransforms'],
    verbose=True
)
# Mettre un pt dans la array de explore3Darray
# explore_3D_array_with_mask_contour(
#     arr=registered_img_ants.numpy(),
#     mask=registered_mask_img_ants.numpy()
# )
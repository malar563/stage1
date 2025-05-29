import ants
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np



# TUTORIEL FINI
# img = ants.image_read('nifti/6_cow_angio__06__hv36__3.nii.gz')
# img2 = ants.image_read('nifti/301_carotid_angio_0625mm.nii.gz').numpy()
# template = ants.image_read("MNI152_T1_1mm.nii.gz")
# print(template)


# Début tutoo
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

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





raw_img_ants = ants.image_read('nifti/2/brain2.nii', reorient='IAL')
explore_3D_array(arr=raw_img_ants.numpy())

template_img_ants = ants.image_read('nifti/miplab-ncct_sym_brain.nii.gz', reorient='IAL')
explore_3D_array(arr = template_img_ants.numpy())

# Image properties
print('\t\tRAW IMG')
print(raw_img_ants)
print('\t\tTEMPLATE IMG')
print(template_img_ants)

transformation = ants.registration(
    fixed=template_img_ants,
    moving=raw_img_ants, 
    type_of_transform='SyN',
    verbose=True
)

print(transformation)

registered_img_ants = transformation['warpedmovout']
explore_3D_array(arr=registered_img_ants.numpy())




# ants.plot(img, overlay = img > img.mean())

# # Weird, marche pas mm si c'est comme dans le tutoriel
# img = ants.smooth_image(img, 2)
# plt.imshow(img[250,:,:], origin="lower")
# plt.show()
# plt.imshow(img[:,250,:], origin="lower")
# plt.show()
# plt.imshow(img[:,:,700], origin="lower")
# plt.show()
# img = ants.resample_image(img, (3,3,3))
# plt.imshow(img[250,:,:], origin="lower")
# plt.show()
# plt.imshow(img[:,250,:], origin="lower")
# plt.show()
# plt.imshow(img[:,:,700], origin="lower")
# plt.show()

# print(img2)

# plt.imshow(img2[250,:,:], origin="lower")
# plt.show()
# plt.imshow(img2[:,250,:], origin="lower")
# plt.show()
# plt.imshow(img2[:,:,700], origin="lower")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure, binary_erosion



def show_3D_array(arr, axis=0, pt=None):
    
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







mask_img = nib.load("jspakoi/0/mask0.nii").get_fdata()
segmentator_img = nib.load("jspakoi/0/totalsegmentator0.nii").get_fdata()

skull_img = np.where(segmentator_img == 91, 1, 0)
brain_img = np.where(segmentator_img ==90, 1, 0)

# Segmenter oreilles
# Ne semble pas fonctionner tel que désiré.
total_img = skull_img + brain_img
dilated_mask = ~binary_dilation(total_img, structure=generate_binary_structure(3, 1), iterations=5)
# show_3D_array(dilated_mask)
# show_3D_array(dilated_mask*mask_img)

# Améliorer skull
mask_head_1 = np.where(mask_img == 1, 1, 0)
# On ne veut rien faire si la zone = 0 ou = 90 dans segmentator : c'est le cas de skull_img
eroded_skull_img = binary_erosion(skull_img, structure=generate_binary_structure(3, 1), iterations=3)
# show_3D_array(eroded_skull_img, axis=2)
# On multiplie le masque du crâne par les tissus mous (incluant les mauvais tissus aka les os).
not_included_skull = 3 * mask_head_1 * eroded_skull_img
show_3D_array(not_included_skull, axis=2)
new_mask_img = mask_img + not_included_skull
show_3D_array(new_mask_img, axis=2)




from segmentation import Segmentation


def to_head(all0 = "jsp"):
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    # Brain mask into brain
    # mask_img = nib.load("nifti/2/totalsegmentator2.nii")
    # segm_img = nib.load("nifti/1/totalsegmentator1.nii")
    mask_array = np.fliplr((np.transpose(ct_scan.head, (2, 1, 0)))) # Mon code donne (z,x,y) ou (z,y,x) à transférer en (x,y,z)

    # head_img =  nib.load(input_img_path)
    head_img =  nib.load("nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz")
    # head_img =  nib.load("nifti/1/cropped_301_carotid_angio_0625mm.nii.gz")
    head_array = head_img.get_fdata()
    head = np.where(mask_array == 1, head_array, -1000) # Put -1000 where the mask is 0



    plt.imshow(head[:,:,100], origin="lower", cmap="gist_gray") # y, x, z
    plt.show()
    plt.imshow(head[:,256,:], origin="lower", cmap="gist_gray") # y, x, z
    plt.show()
    plt.imshow(head[256,:,:], origin="lower", cmap="gist_gray") # y, x, z
    plt.show()

    # Create a new NIfTI image
    head_image = nib.Nifti1Image(head, head_img.affine, head_img.header)

    # Save the new NIfTI image under the same path
    # nifti_path = output_path 
    nifti_path = "nifti/2/head2"
    # nifti_path = "nifti/1/brain1"
    nib.save(head_image, nifti_path)
    print(f"NIfTI generated : {nifti_path}")









# Segmentation
ct_scan = Segmentation() # no.2
# ct_scan = Segmentation(folder_path="DICOM_003/Carotid_Angio_0.625mm") # no.1
print("Resolution", ct_scan.resolution, ct_scan.px_spacing)
ct_scan.cut()
print("Volume shape", ct_scan.array.shape)

ct_scan.apply_threshold()
ct_scan.keep_largest_island()

# ya une ligne qui touche où le nez pour ct_scan.head qui ne s'en va pas (sur 3D slicer non plus)
ct_scan.show(ct_scan.head, 256, "y")
ct_scan.fill_holes()
to_head()
ct_scan.show(ct_scan.head, 256, "y")
# ct_scan.animation(ct_scan.skull)
ct_scan.remove_arteries()
ct_scan.fill_holes()
# ct_scan.show(ct_scan.skull, 256, "y")
# ct_scan.animation(ct_scan.skull)


# Trouver un moyen de ne pas toujours avoir besoin de re-rouler "segmentation" pour avoir "head"
head = ct_scan.head
ct_scan.save_to_pickle(file_name="head")

# ct_scan.show(head, 256, "y")



# Si avec ants :
# 1 - Convertir tous les fichiers dicom en .nii avec nifti.py
# 2 - 



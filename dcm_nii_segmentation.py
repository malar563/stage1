import dicom2nifti
import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np


def dcm_to_nifti(dicom_directory = "DICOM_003/Carotid_Angio_0.625mm", output_directory = "nifti", crop="ouioui"):

    # Create the output_directory file
    os.makedirs(output_directory, exist_ok=True)

    # Convert DICOM to NIfTI (compression=False -> .nii instead of .nii.gz)
    dicom2nifti.convert_directory(dicom_directory, output_directory, compression=True)

    # Find the generated file in the output file
    nifti_files = [f for f in os.listdir(output_directory) if f.endswith('.nii.gz')]

    # Use the first generated file
    nifti_path = os.path.join(output_directory, nifti_files[0])
    print(f"NIfTI generated : {nifti_path}")

    if crop is not None:
        # Load the image with nibabel
        nifti_image = nib.load(nifti_path)

        header = nifti_image.header
        pix_dim, pix_z = header["pixdim"][1:4], header["pixdim"][3]

        # Crop the image
        if pix_z >= 0.6:
            cropped_data = nifti_image.get_fdata()[:,:,-256:]
        else:
            cropped_data = nifti_image.get_fdata()[:,:,-512:]

        # Create a new NIfTI image
        cropped_image = nib.Nifti1Image(cropped_data, nifti_image.affine, nifti_image.header)

        # Save the new NIfTI image under the same path + "cropped"
        nifti_path = os.path.join(output_directory, "cropped_"+nifti_files[0])
        nib.save(cropped_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}")

        shape = cropped_image.shape
        header = cropped_image.header
        affine = cropped_image.affine
        data = cropped_image.get_fdata()

        print("Dimensions :", shape)
        print("Pixel dimensions :", pix_dim)
        # print("Entête :", header)
        # print("Eaffine :", affine)
        # print("data :", data)
        return "cropped_"+nifti_files[0]

# # Rendre plus automatique : si je ne mets pas nifti/1 et /2, ils s'overwritent
# dcm_to_nifti(dicom_directory = "DICOM_003/Carotid_Angio_0.625mm", output_directory = "nifti/1")
# dcm_to_nifti(dicom_directory = "DICOM_010/COW_Angio_0.6_Hv36_3", output_directory = "nifti/2")


def segment(input_path="nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz", output_path="test", fast=False, only_brain=False):
    if __name__ == "__main__":

        input_img = nib.load(input_path)
        if only_brain:
            output_img = totalsegmentator(input_img, fast=fast, roi_subset=["brain"]) # Mettre False pour plus vite
        else:
            output_img = totalsegmentator(input_img, fast=fast) # Mettre False pour plus vite
        nib.save(output_img, output_path)
        # Brain is labeled with the number 90
        # Skull is labeled with the number 91
        
# segment(input_path="nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz", output_path="nifti/2/totalsegmentator2")
# segment(input_path="nifti/1/cropped_301_carotid_angio_0625mm.nii.gz", output_path="nifti/1/totalsegmentator1")



# TROUVER LE MOYEN DE FAIRE MARCHER ÇA AUTOMATIQUE
dicoms_list = ["DICOM_003/Carotid_Angio_0.625mm", "DICOM_010/COW_Angio_0.6_Hv36_3"]

def automatic(dicoms_list = dicoms_list, filename4niftis = "testtii"):
    for i, file_path in enumerate(dicoms_list):
        file = dcm_to_nifti(dicom_directory = file_path, output_directory = filename4niftis + f"/{i+1}")
        segment(input_path = filename4niftis+f"/{i+1}/"+file, output_path = filename4niftis+f"/{i+1}/"+"totalsegmented"+f"{i+1}")

# automatic()


def to_brain(rien="allo"):
    # Brain mask into brain
    # segm_img = nib.load("nifti/2/totalsegmentator2.nii")
    segm_img = nib.load("nifti/1/totalsegmentator1.nii")
    segm_array = segm_img.get_fdata()

    # Keeping the brain
    segm_array = segm_array == 90.0
    segm_array = np.where(segm_array, 1, 0)

    # head_img =  nib.load("nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz")
    head_img =  nib.load("nifti/1/cropped_301_carotid_angio_0625mm.nii.gz")
    head_array = head_img.get_fdata()
    brain = head_array*segm_array

    plt.imshow(brain[:,:,200], origin="lower", cmap="gist_gray") # y, x, z
    plt.show()

    # Create a new NIfTI image
    brain_image = nib.Nifti1Image(brain, head_img.affine, head_img.header)

    # Save the new NIfTI image under the same path 
    # nifti_path = "nifti/2/brain2"
    nifti_path = "nifti/1/brain1"
    nib.save(brain_image, nifti_path)
    print(f"NIfTI generated : {nifti_path}")

to_brain()




import dicom2nifti
import nibabel as nib
import os



def dcm_to_nifti(dicom_directory = "DICOM_003/Carotid_Angio_0.625mm", output_directory = "nifti"):
# def dcm_to_nifti(dicom_directory = "DICOM_010/COW_Angio_0.6_Hv36_3", output_directory = "nifti"):
    
    # Create the output_directory file
    os.makedirs(output_directory, exist_ok=True)

    # Convert DICOM to NIfTI (compression=False -> .nii instead of .nii.gz)
    dicom2nifti.convert_directory(dicom_directory, output_directory, compression=True)

    # Find the  generated file in the output file
    nifti_files = [f for f in os.listdir(output_directory) if f.endswith('.nii.gz')]

    # # Use the first generated file
    nifti_path = os.path.join(output_directory, nifti_files[0])
    print(f"NIfTI généré : {nifti_path}")

    # Load the image with nibabel
    nifti_image = nib.load(nifti_path)

    # Crop the image
    cropped_data = nifti_image.get_fdata()[:,:,-512:]

    # Create a new NIfTI image
    cropped_image = nib.Nifti1Image(cropped_data, nifti_image.affine, nifti_image.header)
    # Il reste juste à le faire enregistrer à la bonne place

    # # Save the new NIfTI image
    nib.save(cropped_image, nifti_files[0])

    shape = nifti_image.shape
    header = nifti_image.header
    affine = nifti_image.affine
    data = nifti_image.get_fdata()
     # 3D array

    # print("Dimensions :", shape)
    # print("Entête :", header)
    # print("Eaffine :", affine)
    # print("data :", data)
    return data
 

print(dcm_to_nifti())

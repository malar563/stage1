import dicom2nifti
import nibabel as nib
import os
# import ants


# def dcm_to_nifti(dicom_directory = "DICOM_003/Carotid_Angio_0.625mm", output_directory = "nifti"):
def dcm_to_nifti(dicom_directory = "DICOM_010/COW_Angio_0.6_Hv36_3", output_directory = "nifti"):
    
    # Crée le fichier demandé
    os.makedirs(output_directory, exist_ok=True)

    # Convertit les fichiers DICOM vers NIfTI (compression=False -> .nii au lieu de .nii.gz)
    dicom2nifti.convert_directory(dicom_directory, output_directory, compression=False)

    # Trouve le fichier généré dans le dossier de sortie
    nifti_files = [f for f in os.listdir(output_directory) if f.endswith('.nii')]

    # Utilise le premier fichier généré
    nifti_path = os.path.join(output_directory, nifti_files[0])
    print(f"NIfTI généré : {nifti_path}")

    # Charge l'image avec nibabel
    nifti_image = nib.load(nifti_path)
    shape = nifti_image.shape
    header = nifti_image.header
    affine = nifti_image.affine
    data = nifti_image.get_fdata() # 3D array

    # print("Dimensions :", shape)
    # print("Entête :", header)
    # print("Eaffine :", affine)
    # print("data :", data)
    return data
 

# print(dcm_to_nifti())

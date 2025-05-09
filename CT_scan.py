import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
# import scikit
# import nibabel as nib


# Trouver un moyen de mettre le nom de fichier automatique le range automatique selon ce que le fichier contient (mettre des try/except)
def load_file():
    dcms = []
    for i in range(4, 605):
        path = r"DICOM_010\COW_Angio_0.6_Hv36_3" + f"\I{i}.dcm"
        dcm_file = dicom.dcmread(path)
        image = dcm_file.pixel_array.astype(np.int16)
        intercept = float(dcm_file.RescaleIntercept)
        slope = float(dcm_file.RescaleSlope)
        dcms.append(image * slope + intercept)
    # print(dcm_file)
    # (0018,0050) Slice Thickness                     DS: '0.6'
    resolution = dcm_file[0x0018, 0x0050].value
    return np.array(dcms), resolution


ct_scan, resolution = load_file()

# print(resolution)
# print(ct_scan)

# # Si l'axe x passe à travers le nasion et règle de la main droite
# # Plan axial : Valeur fixe de z
# plt.imshow(ct_scan[300,:,:], cmap='gist_gray', origin="lower")
# # Plan coronal : Valeur fixe de x
# plt.imshow(ct_scan[:,300,:], cmap='gist_gray', origin="lower")
# # Plan coronal : Valeur fixe de y
plt.imshow(ct_scan[:,:,300], cmap='gist_gray', origin="lower")
plt.show()



def apply_threshold(ct_scan, threshold=-200):
    # Array with "True" where it is, and "False" where it is not
    thresholded_scan = ct_scan >= threshold
    # Put the value of the threshold if True, and -1000 if False
    mask = np.where(thresholded_scan, threshold, -1000)
    return mask

binary_mask = apply_threshold(ct_scan, -200)

plt.imshow(binary_mask[:,:,300], cmap='gist_gray', origin="lower")
# plt.imshow(binary_mask[:,300,:], cmap='gist_gray', origin="lower")
plt.show()



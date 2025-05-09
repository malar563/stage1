import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
# import scikit


# Trouver un moyen de mettre le nom de fichier automatique le range automatique selon ce que le fichier contient
def load_file():
    dcms = []
    for i in range(4, 605):
        path = r"DICOM_010\COW_Angio_0.6_Hv36_3" + f"\I{i}.dcm"
        dcm_file = dicom.dcmread(path)
        dcms.append(dcm_file.pixel_array)
    return np.array(dcms)

image = load_file()

print(image)

# # Si l'axe x passe à travers le nasion et règle de la main droite
# # Plan axial : Valeur fixe de z
# plt.imshow(image[300,:,:], cmap='gist_gray', origin="lower")
# # Plan coronal : Valeur fixe de x
# plt.imshow(image[:,300,:], cmap='gist_gray', origin="lower")
# # Plan coronal : Valeur fixe de y
plt.imshow(image[:,:,300], cmap='gist_gray', origin="lower")
plt.show()


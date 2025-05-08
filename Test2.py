import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
# import scikit


def load_file():
    return dicom.dcmread(r"DICOM_010\COW_Angio_0.6_Hv36_3")

image = load_file()

print(image)
# plt.imshow(image, cmap='gist_gray')
# plt.show()
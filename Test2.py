import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
# import scikit


def load_file(file_number=10):
    return dicom.dcmread(f"DICOM_0{file_number}\COW_Angio_0.6_Hv36_3")

image = load_file()

print(image)
# plt.imshow(image, cmap='gist_gray')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

mask_img = nib.load("jspakoi/0/mask0.nii").get_fdata()
segmentator_img = np.where(nib.load("jspakoi/0/mask0.nii").get_fdata(), 1, 0)


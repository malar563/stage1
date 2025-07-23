import os
import nibabel as nib
import pandas as pd
from class_identification import Identification


# To check the registration and the head
id = Identification(big_output_directory="cava", file_number=1, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
id.show_3D_array(id.head, axis=0)


# def load_landmarks_from_csv(csv_path):
#     """Load and separate MRI and CT landmarks from a CSV file."""
#     df = pd.read_csv(csv_path, sep=",")
    
#     # Get MRI landmarks
#     array_MRI = df.iloc[6:12, 1:].values.astype(float)
#     reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI = array_MRI[1], array_MRI[3], array_MRI[5]
#     imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI = array_MRI[0], array_MRI[2], array_MRI[4]
    
#     # Get CT landmarks
#     array_CT = df.iloc[13:19, 1:].values.astype(float)
#     reg_nas_CT, reg_lpa_CT, reg_rpa_CT = array_CT[1], array_CT[3], array_CT[5]
#     imp_nas_CT, imp_lpa_CT, imp_rpa_CT = array_CT[0], array_CT[2], array_CT[4]

#     return {"reg": {"MRI": [reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI],
#                     "CT":  [reg_nas_CT,  reg_lpa_CT,  reg_rpa_CT]},
#             "imp": {"MRI": [imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI],
#                     "CT":  [imp_nas_CT,  imp_lpa_CT,  imp_rpa_CT]}}




csv_path = os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv")
df = pd.read_csv(csv_path, sep=",")

array_MRI = df.iloc[6:12,1:].values.astype(float)
reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI = array_MRI[1], array_MRI[3], array_MRI[5]
imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI = array_MRI[0], array_MRI[2], array_MRI[4]
print(array_MRI)

array_CT = df.iloc[13:19,1:].values.astype(float)
reg_nas_CT, reg_lpa_CT, reg_rpa_CT = array_CT[1], array_CT[3], array_CT[5]
imp_nas_CT, imp_lpa_CT, imp_rpa_CT = array_CT[0], array_CT[2], array_CT[4]
print(array_CT)

head = nib.load(id.moving_img_path)
head = head.get_fdata()

# OK to change lpa, rpa nas by other values depending on what you want to see
lpa = reg_lpa_CT
rpa = reg_rpa_CT
nas = reg_nas_CT
id.show_3D_array(head, axis=0, pts=[ ((lpa[2],lpa[0]),lpa[1],"blue"),  ((rpa[2],rpa[0]),rpa[1],"red"), ((nas[2],nas[0]),nas[1],"green")])
id.show_3D_array(head, axis=1, pts=[ ((lpa[2],lpa[1]),lpa[0],"blue"),  ((rpa[2],rpa[1]),rpa[0],"red"), ((nas[2],nas[1]),nas[0],"green")])

# Comment the name to keep what you want to see
nifti_img = "totalsegmentator"
nifti_img = "mask"
nifti_img = "mca_territory"
img = nib.load(os.path.join(id.nifti_output_directory, nifti_img+id.file_number+".nii.gz"))

nifti_img = "6_cow_angio__06__hv36__3"
img = nib.load(os.path.join(id.nifti_output_directory, nifti_img+".nii.gz"))

img = img.get_fdata()
id.show_3D_array(img, axis=2)


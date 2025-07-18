import os
import time
import nibabel as nib
import numpy as np
import pandas as pd






from class_identification import Identification
# for i, nifti in enumerate(dicoms_list):

#     id = Registration(big_output_directory="online", file_number=i, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

#     # id.register(show=True)
#     id.read_transforms()

#     id.find_registered_lpa_rpa_nasion()
#     id.fill_cavities()
#     id.find_nasion()
#     id.check_nasion()
#     id.find_rpa()
#     print("rpa :", id.rpa)
#     id.show_3D_array(id.head, axis=2, pt=(id.rpa[1], id.rpa[0]), pt_slice=id.rpa[2])
#     id.find_lpa()
#     print("lpa :", id.lpa)
#     id.show_3D_array(id.head, axis=2, pt=(id.lpa[1], id.lpa[0]), pt_slice=id.lpa[2])

#     # id.save_pts_to_csv()
#     # id.mca_territory_mask()




# AVANT CETAIT PAS EN COMMENTAIRE ->

# # 6_cta_thins # online_angio no.2
id = Identification(big_output_directory="50_2025-07-17", file_number=8, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')#50_p_2025-07-15
id.show_3D_array(id.head, axis=0)


# # id.register(show=True)
# id.read_transforms()

# id.find_registered_lpa_rpa_nasion()
# id.fill_cavities()
# id.find_nasion(window=5)
# # id.check_nasion()
# id.improve_lpa_rpa()
# print("rpa :", id.rpa)
# print("registered_rpa :", id.registered_rpa)
# id.show_3D_array(id.head, axis=2, pts=[((id.rpa[1], id.rpa[0]),id.rpa[2], "red"), ((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "green")])
# # id.show_3D_array(id.head, axis=2, pt=(id.registered_rpa[1], id.registered_rpa[0]), pt_slice=id.registered_rpa[2])
# print("lpa :", id.lpa)
# print("registered_rpa :", id.registered_lpa)
# id.show_3D_array(id.head, axis=2, pts=[((id.lpa[1], id.lpa[0]),id.lpa[2], "red"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])
# # id.show_3D_array(id.head, axis=2, pt=(id.lpa[1], id.lpa[0]), pt_slice=id.lpa[2])
# # id.show_3D_array(id.head, axis=2, pt=(id.registered_lpa[1], id.registered_lpa[0]), pt_slice=id.registered_lpa[2])

# # id.delete_useless_files()
# id.save_pts_to_csv()
# # # id.mca_territory_mask()







# Check if the registration is fine
csv_path = os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv")
df = pd.read_csv(csv_path, sep=",")

array_MRI = df.iloc[6:12,1:].values.astype(float)
print(array_MRI)
reg_nas_MRI = array_MRI[1]
reg_lpa_MRI = array_MRI[3]
reg_rpa_MRI = array_MRI[5]
print(reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI)

img = nib.load(id.moving_img_path)
img = img.get_fdata()
id.show_3D_array(img, axis=0, pts=[ ((reg_lpa_MRI[2],reg_lpa_MRI[0]),reg_lpa_MRI[1],"blue"),  ((reg_rpa_MRI[2],reg_rpa_MRI[0]),reg_rpa_MRI[1],"red")])
id.show_3D_array(img, axis=1, pts=[ ((reg_nas_MRI[2],reg_nas_MRI[1]),reg_nas_MRI[0],"red") ])
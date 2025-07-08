import os
import time
import nibabel as nib
import numpy as np
import pandas as pd


    # # dicoms_list = ["DICOM_003/Carotid_Angio_0.625mm", "DICOM_010/COW_Angio_0.6_Hv36_3"]
    # dicoms_list = ["online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463", "online_patient/2.16.840.1.114274.1818.528945204283203896414435929150802789774", "online_patient/2.16.840.1.114274.1818.56920369040074765021783555636978216368"]
    # # dicoms_list = ["online_patient/test", "2.16.840.1.114274.1818.528945204283203896414435929150802789774", "2.16.840.1.114274.1818.56920369040074765021783555636978216368"]
    # dicoms_list = ["ct_enligne/1", "ct_enligne/2","ct_enligne/4","ct_enligne/5", "ct_enligne/6", "ct_enligne/7"]






from identification import Registration
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



# 6_cta_thins # online_angio no.2
id = Registration(big_output_directory="online_angio", file_number=2, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

# id.register(show=True)
id.read_transforms()

id.find_registered_lpa_rpa_nasion()
id.fill_cavities()
id.find_nasion(window=5)
# id.check_nasion()
id.improve_lpa_rpa()
print("rpa :", id.rpa)
print("registered_rpa :", id.registered_rpa)
id.show_3D_array(id.head, axis=2, pts=[((id.rpa[1], id.rpa[0]),id.rpa[2], "red"), ((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "green")])
# id.show_3D_array(id.head, axis=2, pt=(id.registered_rpa[1], id.registered_rpa[0]), pt_slice=id.registered_rpa[2])
print("lpa :", id.lpa)
print("registered_rpa :", id.registered_lpa)
id.show_3D_array(id.head, axis=2, pts=[((id.lpa[1], id.lpa[0]),id.lpa[2], "red"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])
# id.show_3D_array(id.head, axis=2, pt=(id.lpa[1], id.lpa[0]), pt_slice=id.lpa[2])
# id.show_3D_array(id.head, axis=2, pt=(id.registered_lpa[1], id.registered_lpa[0]), pt_slice=id.registered_lpa[2])

# id.delete_useless_files()
id.save_pts_to_csv()
# # id.mca_territory_mask()





# Check if the registration is fine
csv_path = os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv")
df = pd.read_csv(csv_path, sep=",")
array = df.iloc[5:11,1:].values.astype(float)
print(array)
reg_nas = array[1]
reg_lpa = array[3]
reg_rpa = array[5]
print(reg_nas, reg_lpa, reg_rpa)

img = nib.load(id.moving_img_path)
img = img.get_fdata()
id.show_3D_array(img, axis=0, pts=[ ((reg_lpa[2],reg_lpa[0]),reg_lpa[1],"blue"),  ((reg_rpa[2],reg_rpa[0]),reg_rpa[1],"red")])
id.show_3D_array(img, axis=1, pts=[ ((reg_nas[2],reg_nas[1]),reg_nas[0],"red") ])
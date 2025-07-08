import os
import time
from segmentation import Segmentation
from identification import Identification


                

def run_everything(dicoms_list):

    for i, dicom in enumerate(dicoms_list):
        try:
            start = time.time()
            ct = Segmentation(dcm_path=dicom, big_output_directory="tout_change", file_number=i+4)
            ct.apply_threshold()       
            ct.keep_largest_island()
            ct.show_3D_array(ct.skull, axis=2) 
            ct.fill_holes()
            ct.remove_arteries()

            # Totalsegmentator
            ct.segment_brain()
            ct.arteries_and_totalsegmentator_mask()
            ct.mask_to_nii()

            # ct.show_3D_array(ct.skull, axis=2) # En z
            # ct.show_3D_array(ct.head, axis=0, pt=(50,42), pt_slice = 100)
            id = Identification(big_output_directory="ct_enligne_nifti", file_number=i+4, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

            id.register(show=True)
            # id.read_transforms()   

            id.find_registered_lpa_rpa_nasion()
            id.fill_cavities()
            id.find_nasion()
            id.check_nasion()
            id.find_rpa()
            print("rpa :", id.rpa)
            print("registered_rpa :", id.registered_rpa)
            id.show_3D_array(id.head, axis=2, pt=(id.rpa[1], id.rpa[0]), pt_slice=id.rpa[2])
            id.show_3D_array(id.head, axis=2, pt=(id.registered_rpa[1], id.registered_rpa[0]), pt_slice=id.registered_rpa[2])
            id.find_lpa()
            print("lpa :", id.lpa)
            print("registered_rpa :", id.registered_lpa)
            id.show_3D_array(id.head, axis=2, pt=(id.lpa[1], id.lpa[0]), pt_slice=id.lpa[2])
            id.show_3D_array(id.head, axis=2, pt=(id.registered_lpa[1], id.registered_lpa[0]), pt_slice=id.registered_lpa[2])

            print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")
        except Exception as e:
            with open(os.path.join(ct.big_output_directory, 'error.txt'), 'a') as file:
                file.write(f"{ct.nii_path} : {e}\n")


if __name__ == "__main__":

    from automatically_get_dicom_folders import get_dicom_folders

    dicoms_list = ["ct_enligne/1", "ct_enligne/2","ct_enligne/4","ct_enligne/5", "ct_enligne/6", "ct_enligne/7"]
    # dicoms_list = get_dicom_folders(directory="nifti")

    run_everything(dicoms_list=dicoms_list)









    # start = time.time()
    # ct = Segmentation(dcm_path="online_patient/2.16.840.1.114274.1818.56920369040074765021783555636978216368", big_output_directory="online", file_number=2)
    # ct.apply_threshold()       
    # ct.keep_largest_island()
    # ct.fill_holes()
    # ct.remove_arteries()

    # # Totalsegmentator
    # # ct.segment_brain()
    # ct.arteries_and_totalsegmentator_mask()
    # ct.mask_to_nii()

    # print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")














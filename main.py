import os
import time
from class_segmentation import Segmentation
from class_identification import Identification


                

def run_everything(dicoms_list, big_output_directory="cava", verbose=True):

    for i, dicom in enumerate(dicoms_list):
        try:
            start = time.time()

            ct = Segmentation(dcm_path=dicom, big_output_directory=big_output_directory, file_number=i)

            ct.apply_threshold()
            if verbose:
                print("Thresholds applied")

            ct.keep_largest_island()
            if verbose:
                print("Kept largest island")

            # ct.show_3D_array(ct.skull, axis=2) 
            ct.fill_holes()
            if verbose:
                print("Holes filled")

            ct.remove_arteries()
            if verbose:
                print("Distance arteries removed")
            # Totalsegmentator
            ct.segment_brain()
            if verbose:
                print("Head segmented (TotalSegmentator)")

            ct.arteries_and_totalsegmentator_mask()
            if verbose:
                print("TotalSegmentator masks created")

            ct.mask_to_nii()
            if verbose:
                print(f"Segmentation complete: {ct.nii_path}")

            # # ct.show_3D_array(ct.skull, axis=2) # En z
            # # ct.show_3D_array(ct.head, axis=0, pt=(50,42), pt_slice = 100)

            id = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path="head1.nii.gz", register_with_CT_not_normalized=True)#"head1.nii.gz""cropped_605_sag_1mm.nii.gz" 'icbm_avg_152_t1_tal_lin.nii'
            # id = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path='icbm_avg_152_t1_tal_lin.nii', register_with_CT_not_normalized=False)

            id.register(show=True)
            if verbose:
                print("Registration done")

            # id.read_transforms()
            # if verbose:
            #     print("Transforms loaded")

            id.find_registered_lpa_rpa_nasion()
            if verbose:
                print("Registered LPA, RPA, and Nasion located")
            print("registered_rpa :", id.registered_rpa)
            print("registered_rpa :", id.registered_lpa)
            id.show_3D_array(id.head, axis=2, pts=[((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "red"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])

            id.find_nasion()
            if verbose:
                print("Nasion refined")
            id.check_nasion()

            id.improve_lpa_rpa()
            if verbose:
                print("LPA and RPA refined")

            print("rpa :", id.rpa)
            print("lpa :", id.lpa)
            print("registered_rpa :", id.registered_rpa)
            print("registered_rpa :", id.registered_lpa)
            id.show_3D_array(id.head, axis=2, pts=[((id.rpa[1], id.rpa[0]),id.rpa[2], "red"), ((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "green"), ((id.lpa[1], id.lpa[0]),id.lpa[2], "blue"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])
            

            print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")

            with open(os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv"),'a') as fd:
                processing_time = f"Processing time (seconds), {time.time()-start}, for file, {ct.nii_path}"
                fd.write(processing_time)

        except Exception as e:
            with open(os.path.join(ct.big_output_directory, 'error.txt'), 'a') as file:
                file.write(f"{ct.nii_path} : {e}\n")


if __name__ == "__main__":

    from automatically_get_dicom_folders import create_dicom_list

    dicoms_list = ["ct_enligne/1", "ct_enligne/2"]
    # dicoms_list = create_dicom_list(directory="nifti")

    run_everything(dicoms_list=dicoms_list, big_output_directory="reg_sans_irm")







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














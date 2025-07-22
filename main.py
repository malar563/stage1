import os
import time
from class_segmentation import Segmentation
from class_identification import Identification


def run_everything(dicoms_list, big_output_directory="processed_files", verbose=True, read=False, register_with_MRI=True, register_with_CT=False):

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
 
            ct.fill_holes()
            if verbose:
                print("Holes filled")

            ct.remove_arteries()
            if verbose:
                print("Distance arteries removed")
            
            ct.segment_brain() # Totalsegmentator
            if verbose:
                print("Head segmented (TotalSegmentator)")

            ct.arteries_and_totalsegmentator_mask()
            if verbose:
                print("TotalSegmentator masks created")

            ct.mask_to_nii()
            if verbose:
                print(f"Segmentation complete: {ct.nii_path}")

            segmentation_processing_time = time.time()-start

            # ct.show_3D_array(ct.skull, axis=2) # z-axis

            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # Registration with a CT scan not normalized

            if register_with_CT:
                id = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path="head1.nii.gz", register_with_CT_not_normalized=True)

                if not read:
                    id.register(show=False)
                    if verbose:
                        print("Registration done")

                if read:
                    id.read_transforms()
                    if verbose:
                        print("Transforms loaded")

                id.find_registered_lpa_rpa_nasion()
                if verbose:
                    print("Registered LPA, RPA, and Nasion located")
                # id.show_3D_array(id.head, axis=2, pts=[((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "red"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])

                id.find_nasion()
                if verbose:
                    print("Nasion refined")
                id.check_nasion()

                id.improve_lpa_rpa()
                if verbose:
                    print("LPA and RPA refined")
                # id.show_3D_array(id.head, axis=2, pts=[((id.rpa[1], id.rpa[0]),id.rpa[2], "red"), ((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "green"), ((id.lpa[1], id.lpa[0]),id.lpa[2], "blue"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])
                
                id.save_pts_to_csv()

                print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")

                with open(os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv"),'a') as fd:
                    id_ct_processing_time = f"CT processing time (seconds), {time.time()-start}, for file, {ct.nii_path}"
                    fd.write(id_ct_processing_time)

            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # Registration with an MRI normalized scan

            if register_with_MRI:
                id = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

                if not read:
                    id.register(show=False)
                    if verbose:
                        print("Registration done")

                if read:
                    id.read_transforms()
                    if verbose:
                        print("Transforms loaded")

                id.find_registered_lpa_rpa_nasion()
                if verbose:
                    print("Registered LPA, RPA, and Nasion located")
                # id.show_3D_array(id.head, axis=2, pts=[((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "red"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])

                id.find_nasion()
                if verbose:
                    print("Nasion refined")
                # id.check_nasion()

                id.improve_lpa_rpa()
                if verbose:
                    print("LPA and RPA refined")
                # id.show_3D_array(id.head, axis=2, pts=[((id.rpa[1], id.rpa[0]),id.rpa[2], "red"), ((id.registered_rpa[1], id.registered_rpa[0]),id.registered_rpa[2], "green"), ((id.lpa[1], id.lpa[0]),id.lpa[2], "blue"), ((id.registered_lpa[1], id.registered_lpa[0]),id.registered_lpa[2], "green")])
                
                id.save_pts_to_csv()

                print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")

                with open(os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv"),'a') as fd:
                    mri_processing_time = f"MRI processing time (seconds), {time.time()-start}, for file, {ct.nii_path}"
                    fd.write(mri_processing_time)

            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------

            with open(os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv"),'a') as fd:
                segmentation_processing_time = f"Segmentation processing time (seconds), {segmentation_processing_time}, for file, {ct.nii_path}"
                fd.write(segmentation_processing_time+id_ct_processing_time)


        except Exception as e:
            try:
                with open(os.path.join(ct.big_output_directory, 'error.txt'), 'a') as file:
                    file.write(f"{ct.nii_path} : {e}\n")
            except:
                with open(os.path.join(big_output_directory, 'not_converted.txt'), 'a') as file:
                    file.write(f"{dicom} : {e}\n")                



# ---------- USER SECTION: Only modify parameters below this line ----------
if __name__ == "__main__":

    from automatically_get_dicom_folders import create_list
    # dicoms_list = create_list(directory="50_CQ")
    # print(dicoms_list, len(dicoms_list))


    # dicoms_list = [r"50_CQ\CQ500CT0 CQ500CT0\Unknown Study\CT PLAIN THIN", r"50_CQ\CQ500CT2 CQ500CT2\Unknown Study\CT 0.625mm", r"50_CQ\CQ500CT3 CQ500CT3\Unknown Study\CT PLAIN THIN",
    #                r"50_CQ\CQ500CT4 CQ500CT4\Unknown Study\CT 0.625mm", r"50_CQ\CQ500CT6 CQ500CT6\Unknown Study\CT Thin Details", r"50_CQ\CQ500CT10 CQ500CT10\Unknown Study\CT PLAIN THIN",
    #                 r"50_CQ\CQ500CT17 CQ500CT17\Unknown Study\CT 0.625mm", r"50_CQ\CQ500CT18 CQ500CT18\Unknown Study\CT 0.625mm", r"50_CQ\CQ500CT22 CQ500CT22\Unknown Study\CT PLAIN THIN",
    #                 r"50_CQ\CQ500CT26 CQ500CT26\Unknown Study\CT C THIN", r"50_CQ\CQ500CT32 CQ500CT32\Unknown Study\CT 0.625mm", r"50_CQ\CQ500CT40 CQ500CT40\Unknown Study\CT 0.625mm",
    #                 r"50_CQ\CQ500CT48 CQ500CT48\Unknown Study\CT PLAIN THIN", r"50_CQ\CQ500CT50 CQ500CT50\Unknown Study\CT 0.625mm"]


    dicoms_list = ["150_CQ/CQ500CT55 CQ500CT55/Unknown Study/CT 5mm", "150_CQ/CQ500CT57 CQ500CT57/Unknown Study/CT 0.625mm",
                   "150_CQ/CQ500CT60 CQ500CT60/Unknown Study/CT 0.625mm", "150_CQ/CQ500CT66 CQ500CT66/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT67 CQ500CT67/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT73 CQ500CT73/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT78 CQ500CT78/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT80 CQ500CT80/Unknown Study/CT 0.625mm",
                   "150_CQ/CQ500CT84 CQ500CT84/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT85 CQ500CT85/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT90 CQ500CT90/Unknown Study/CT 0.625mm", "150_CQ/CQ500CT92 CQ500CT92/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT101 CQ500CT101/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT102 CQ500CT102/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT104 CQ500CT104/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT108 CQ500CT108/Unknown Study/CT 0.625mm",
                   "150_CQ/CQ500CT109 CQ500CT109/Unknown Study/CT 0.625mm", "150_CQ/CQ500CT111 CQ500CT111/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT113 CQ500CT113/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT121 CQ500CT121/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT126 CQ500CT126/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT130 CQ500CT130/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT135 CQ500CT135/Unknown Study/CT PLAIN THIN", "150_CQ/CQ500CT140 CQ500CT140/Unknown Study/CT PLAIN THIN",
                   "150_CQ/CQ500CT149 CQ500CT149/Unknown Study/CT 0.625mm"]

    run_everything(dicoms_list=dicoms_list, big_output_directory="150_2025-07-21", verbose=True, read=False, register_with_MRI=True, register_with_CT=True)














import os
import time
from tqdm import tqdm

from class_segmentation import Segmentation
from class_identification import Identification


def run_everything(dicoms_list, big_output_directory="processed_files", register_with_MRI=True, register_with_CT=False, read_transforms=False, delete_useless_files=False, verbose=True):

    for i, dicom in tqdm(enumerate(dicoms_list)):
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
            # Registration with an MRI normalized scan

            if register_with_MRI:
                id = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

                if not read_transforms:
                    id.register(show=False)
                    if verbose:
                        print("Registration done")

                if read_transforms:
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
                    mri_processing_time = f"MRI processing time (seconds), {time.time()-start}, for file, {ct.nii_path}\n"
                    fd.write(mri_processing_time)
                    fd.close()

                if delete_useless_files:
                    id.delete_useless_files()
                    if verbose:
                        print("Useless files deleted")


            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # Registration with a CT scan not normalized

            if register_with_CT:
                id_ct = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path="head1.nii.gz", register_with_CT_not_normalized=True)

                if not read_transforms:
                    id_ct.register(show=False)
                    if verbose:
                        print("Registration done")

                if read_transforms:
                    id_ct.read_transforms()
                    if verbose:
                        print("Transforms loaded")

                id_ct.find_registered_lpa_rpa_nasion()
                if verbose:
                    print("Registered LPA, RPA, and Nasion located")
                # id_ct.show_3D_array(id_ct.head, axis=2, pts=[((id_ct.registered_rpa[1], id_ct.registered_rpa[0]),id_ct.registered_rpa[2], "red"), ((id_ct.registered_lpa[1], id_ct.registered_lpa[0]),id_ct.registered_lpa[2], "green")])

                id_ct.find_nasion()
                if verbose:
                    print("Nasion refined")
                # id_ct.check_nasion()

                id_ct.improve_lpa_rpa()
                if verbose:
                    print("LPA and RPA refined")
                # id_ct.show_3D_array(id_ct.head, axis=2, pts=[((id_ct.rpa[1], id_ct.rpa[0]),id_ct.rpa[2], "red"), ((id_ct.registered_rpa[1], id_ct.registered_rpa[0]),id_ct.registered_rpa[2], "green"), ((id_ct.lpa[1], id_ct.lpa[0]),id_ct.lpa[2], "blue"), ((id_ct.registered_lpa[1], id_ct.registered_lpa[0]),id_ct.registered_lpa[2], "green")])
                
                id_ct.save_pts_to_csv()

                print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")

                with open(os.path.join(id_ct.nifti_output_directory, "points"+id_ct.file_number+".csv"),'a') as fd:
                    ct_processing_time = f"CT processing time (seconds), {time.time()-start}, for file, {ct.nii_path}\n"
                    fd.write(ct_processing_time)
                    fd.close()
                
                if delete_useless_files:
                    id_ct.delete_useless_files()
                    if verbose:
                        print("Useless files deleted")


            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            
            if not register_with_MRI:
                with open(os.path.join(ct.nifti_output_directory, "points"+ct.file_number+".csv"),'a') as fd:
                    mri_processing_time = f"MRI processing time (seconds), {0}, for file, {ct.nii_path}\n"
                    fd.write(mri_processing_time)
                    fd.close()
            

            if not register_with_CT:
                with open(os.path.join(ct.nifti_output_directory, "points"+ct.file_number+".csv"),'a') as fd:
                    ct_processing_time = f"CT processing time (seconds), {0}, for file, {ct.nii_path}\n"
                    fd.write(ct_processing_time)
                    fd.close()
            


            with open(os.path.join(ct.nifti_output_directory, "points"+ct.file_number+".csv"),'a') as fd:
                segmentation_processing_time = f"Segmentation processing time (seconds), {segmentation_processing_time}, for file, {ct.nii_path}"
                fd.write(segmentation_processing_time)
                fd.close()


        except Exception as e:
            try:
                with open(os.path.join(ct.big_output_directory, 'error.txt'), 'a') as file:
                    file.write(f"{ct.nii_path} : {e}\n")
            except:
                with open(os.path.join(big_output_directory, 'not_converted.txt'), 'a') as file:
                    file.write(f"{dicom} : {e}\n")                



# ---------- USER SECTION: Only modify parameters below this line ----------
if __name__ == "__main__":

    from automatically_get_folders import create_list
    # dicoms_list = create_list(directory="50_CQ")
    # print(dicoms_list, len(dicoms_list))

    dicoms_list = ["250_CQ/CQ500CT0 CQ500CT0/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT2 CQ500CT2/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT3 CQ500CT3/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT4 CQ500CT4/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT6 CQ500CT6/Unknown Study/CT Thin Details", "250_CQ/CQ500CT10 CQ500CT10/Unknown Study/CT PLAIN THIN",
                    "250_CQ/CQ500CT17 CQ500CT17/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT18 CQ500CT18/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT22 CQ500CT22/Unknown Study/CT PLAIN THIN",
                    "250_CQ/CQ500CT26 CQ500CT26/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT40 CQ500CT40/Unknown Study/CT 0.625mm",
                    "250_CQ/CQ500CT50 CQ500CT50/Unknown Study/CT 0.625mm",
                    "250_CQ/CQ500CT55 CQ500CT55/Unknown Study/CT 0.625mm",
                   "250_CQ/CQ500CT60 CQ500CT60/Unknown Study/CT 0.625mm-2", "250_CQ/CQ500CT66 CQ500CT66/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT67 CQ500CT67/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT78 CQ500CT78/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT84 CQ500CT84/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT85 CQ500CT85/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT92 CQ500CT92/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT102 CQ500CT102/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT108 CQ500CT108/Unknown Study/CT 0.625mm",
                   "250_CQ/CQ500CT109 CQ500CT109/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT111 CQ500CT111/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT113 CQ500CT113/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT121 CQ500CT121/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT126 CQ500CT126/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT130 CQ500CT130/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT135 CQ500CT135/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT140 CQ500CT140/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT149 CQ500CT149/Unknown Study/CT 0.625mm-3",
                   "250_CQ/CQ500CT152 CQ500CT152/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT154 CQ500CT154/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT162 CQ500CT162/Unknown Study/CT PLAIN THIN-2", "250_CQ/CQ500CT166 CQ500CT166/Unknown Study/CT 0.625mm-2",
                   "250_CQ/CQ500CT167 CQ500CT167/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT179 CQ500CT179/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT183 CQ500CT183/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT188 CQ500CT188/Unknown Study/CT 0.625mm-3",
                   "250_CQ/CQ500CT193 CQ500CT193/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT196 CQ500CT196/Unknown Study/CT 0.625mm",
                   "250_CQ/CQ500CT204 CQ500CT204/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT216 CQ500CT216/Unknown Study/CT 0.625mm",
                   "250_CQ/CQ500CT219 CQ500CT219/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT220 CQ500CT220/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT221 CQ500CT221/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT233 CQ500CT233/Unknown Study/CT PLAIN THIN",
                   "250_CQ/CQ500CT237 CQ500CT237/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT241 CQ500CT241/Unknown Study/CT PLAIN THIN-2",
                   "250_CQ/CQ500CT243 CQ500CT243/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT249 CQ500CT249/Unknown Study/CT 0.625mm",
                   "250_CQ/CQ500CT250 CQ500CT250/Unknown Study/CT PLAIN THIN"]

    run_everything(dicoms_list=dicoms_list, big_output_directory="250_2025-07-31", register_with_MRI=True, register_with_CT=True, read_transforms=True, delete_useless_files=False, verbose=True)














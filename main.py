import os
import time
from tqdm import tqdm

from class_segmentation import Segmentation
from class_identification import Identification
from automatically_get_folders import create_list


def run_everything(dicoms_list, big_output_directory="processed_files", register_with_MRI=True, register_with_CT=False, read_transforms=False, delete_useless_files=False, verbose=True):
    """
    Run the full segmentation and registration pipeline on a list of DICOM directories.

    This function processes each DICOM scan by:
    0. Transforming the DICOM folder into a NIfTI file if needed.
    1. Segmenting the head and extracting brain/arterial structures.
    2. Registering the result with a normalized MRI and/or CT scan.
    3. Refining anatomical landmarks (LPA, RPA, Nasion).
    4. Saving outputs and processing times.
    5. Optionally deleting intermediate files to save disk space.

    Parameters
    ----------
    dicoms_list : list of str
        List of paths to directories containing DICOM files for each scan to process.

    big_output_directory : str, optional
        Base directory to save all outputs. Each case is saved in a subfolder.
        Default is "processed_files".

    register_with_MRI : bool, optional
        Whether to perform registration with a normalized MRI space.
        Default is True.

    register_with_CT : bool, optional
        Whether to perform registration to a non-normalized CT space.
        Default is False.

    read_transforms : bool, optional
        If True, reads precomputed registration transforms instead of recomputing them.
        Default is False.

    delete_useless_files : bool, optional
        If True, removes intermediate and temporary files after processing.
        Default is False.

    verbose : bool, optional
        If True, prints progress messages at each stage of processing.
        Default is True.

    Outputs
    -------
    For each input scan:
    - NIfTI equivalent of the input DICOM (cropped or not)
    - Segmented NIfTI file of the head and arteries (head and mask).
    - Anatomical landmarks saved in a CSV file.
    - Processing times recorded in the same CSV file.
    - Forward and inverse transforms from registration.
    - Mask of MC artery.
    - Optional deletion of intermediate files.
    - Errors, if any, logged in `error.txt` or `not_converted.txt` in the output directory.

    Notes
    -----
    - Segmentation is performed using custom logic and the TotalSegmentator tool.
    - Registration is handled using the `Identification` class and requires preconfigured target images.
    - This function is intended for batch processing of multiple scans.
    """
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
                id = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path='icbm_avg_152_t1_tal_lin.nii', register_with_CT_not_normalized=False)

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
                
                id.mca_territory_mask()
                if verbose:
                    print("MCA mask created")

                id.save_pts_to_csv()

                print(f"Time to register with MRI {ct.nii_path} : {time.time() - start} seconds")

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
                id_ct = Identification(big_output_directory=big_output_directory, file_number=i, fixed_img_path="ct_ref.nii.gz", register_with_CT_not_normalized=True) # To change the CT normalized space, change fixed_img_path here 

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

                print(f"Time to register with CT {ct.nii_path} : {time.time() - start} seconds")

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

    dicoms_directory = "50_CQ" # Put your DICOM folder
    # dicoms_list = create_list(directory=dicoms_directory) # Uncomment here if needed
    # print(dicoms_list, len(dicoms_list)) # Uncomment here if needed

    dicoms_list = ["250_CQ/CQ500CT0 CQ500CT0/Unknown Study/CT PLAIN THIN", "250_CQ/CQ500CT2 CQ500CT2/Unknown Study/CT 0.625mm", "250_CQ/CQ500CT3 CQ500CT3/Unknown Study/CT PLAIN THIN"]


    """
    dicoms_list:
        List of paths to DICOM folders to process
        
    big_output_directory:
        Name of the folder that will contain each patient's subfolder
    
    register_with_MRI:
        Whether or not to do the registration with the normalized MRI scan
        
    register_with_CT:
        Whether or not to do the registration with the non-normalized CT scan
    
    read_transforms:
        Only put to True if the registration has already been done and the transform files are still in the folder
        
    delete_useless_files:
        Only put to True if you are sure you won't need files involved anymore. If you have enough space, it might be better to do it after the whole process with delete_files.py
        
    verbose :
        Whether or not you want to see updates of the process"""


    run_everything(dicoms_list=dicoms_list, big_output_directory="processed_files", register_with_MRI=True, register_with_CT=True, read_transforms=False, delete_useless_files=False, verbose=True)

    
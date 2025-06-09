import dicom2nifti
import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np
import ants


class Segmentation:
    
    def __init__(self, dcm_path="DICOM_010/COW_Angio_0.6_Hv36_3", big_output_directory="processed_files", file_number=0):
        self.array = None
        # self.resolution = None
        # self.px_spacing = None
        self.dcm_path = dcm_path
        self.big_output_directory = big_output_directory
        self.file_number = str(file_number)
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number)
        self.nii_path = None
        
        # All of these arrays are masks
        self.no_arteries_array = None
        self.arteries = None
        self.head = None
        self.skull = None
        self.air = None
        self.brain = None
  

    def dcm_to_nii(self, crop="yes"):

        dicom_directory = self.dcm_path

        # Create the output_directory file
        os.makedirs(self.nifti_output_directory, exist_ok=True)

        # Convert DICOM to NIfTI (compression=False -> .nii instead of .nii.gz)
        dicom2nifti.convert_directory(dicom_directory, self.nifti_output_directory, compression=True)

        # Find the generated file in the output file
        nifti_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith('.nii.gz')]

        # Use the first generated file
        nifti_path = os.path.join(self.nifti_output_directory, nifti_files[0])
        print(f"NIfTI generated : {nifti_path}")

        if crop is not None:
            # Load the image with nibabel
            nifti_image = nib.load(nifti_path)

            header = nifti_image.header
            pix_dim, pix_z = header["pixdim"][1:4], header["pixdim"][3]

            # Crop the image
            if pix_z >= 0.6:
                cropped_data = nifti_image.get_fdata()[:,:,-256:]
            else:
                cropped_data = nifti_image.get_fdata()[:,:,-512:]

            # Create a new NIfTI image
            cropped_image = nib.Nifti1Image(cropped_data, nifti_image.affine, nifti_image.header)

            # Save the new NIfTI image under the same path + "cropped"
            nifti_path = os.path.join(self.nifti_output_directory, "cropped_"+nifti_files[0])
            nib.save(cropped_image, nifti_path)
            print(f"NIfTI generated : {nifti_path}")

            shape = cropped_image.shape
            header = cropped_image.header
            affine = cropped_image.affine
            data = cropped_image.get_fdata()

            print("Dimensions :", shape)
            print("Pixel dimensions :", pix_dim)
            # print("Entête :", header)
            # print("Eaffine :", affine)
            # print("data :", data)
            self.nii_path = nifti_path
            return self.nii_path
        
    
    def load_nii(self):
        self.array = nib.load(self.nii_path).get_fdata()
        return self.array
    

    def show_3D_array(self, arr, axis=0, pt=None): # y=0, x=1, z=2
        from matplotlib.widgets import Slider

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # Initial slice index
        index = arr.shape[axis] // 2
        if axis == 0:
            img = ax.imshow(arr[index, :, :], cmap="gray", origin="lower")
        elif axis == 1:
            img = ax.imshow(arr[:, index, :], cmap="gray", origin="lower")
        else:
            img = ax.imshow(arr[:, :, index], cmap="gray", origin="lower")

        # Slider setup
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slice_slider = Slider(ax_slider, 'Slice', 0, arr.shape[axis] - 1, valinit=index, valstep=1)

        def update(val):
            if axis == 0:
                img.set_data(arr[int(slice_slider.val), :, :])
            elif axis == 1:
                img.set_data(arr[:, int(slice_slider.val), :])
            else:
                img.set_data(arr[:, :, int(slice_slider.val)])
            fig.canvas.draw_idle()

        slice_slider.on_changed(update)
        plt.show()


    def apply_threshold(self, threshold_head=-200, threshold_skull=200, threshold_no_arteries = 500, threshold_arteries = 100):
        # Array with "True" where it is, and "False" where it is not
        thresholded_head = self.array >= threshold_head
        thresholded_air = self.array <= threshold_head
        thresholded_skull = self.array >= threshold_skull
        thresholded_no_arteries = self.array >= threshold_no_arteries
        thresholded_arteries = self.array >= threshold_arteries
        # Put the value 1 if True, and 0 if False
        self.head = np.where(thresholded_head, 1, 0)
        self.air = np.where(thresholded_air, 1, 0)
        self.skull = np.where(thresholded_skull, 1, 0)
        self.no_arteries_array = np.where(thresholded_no_arteries, 1, 0)
        self.arteries = np.where(thresholded_arteries, 1, 0)

        return self.head, self.skull, self.no_arteries_array, self.air, self.arteries
    
    
    def keep_largest_island(self):
        from scipy.ndimage import label, generate_binary_structure

        def largest_connected_island(mask):
            s = generate_binary_structure(3, 3)
            labeled, _ = label(mask, s) # Associate a number to an island
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # ignore background
            return labeled == np.argmax(counts) # Index of the maximum count = number given by np.label

        self.head = largest_connected_island(self.head)
        self.skull = largest_connected_island(self.skull)
        self.no_arteries_array = largest_connected_island(self.no_arteries_array)
        self.air = largest_connected_island(self.air != 1)
        self.air = np.where(self.air, 0, 1)

        return self.head, self.skull, self.no_arteries_array, self.air
    

    def fill_holes(self):
        from scipy.ndimage import binary_fill_holes

        self.skull = binary_fill_holes(self.skull)
        return self.skull


    def remove_arteries(self, max_distance = 3): # Mettre 200 et 500 comme seuil avec cette distance
        from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure

        self.no_arteries_array = self.no_arteries_array != 1
        distance = distance_transform_edt(self.no_arteries_array)
        close_to_bone = distance < max_distance
        self.skull = self.skull & close_to_bone
        self.skull = binary_dilation(self.skull, generate_binary_structure(3, 1))

        return self.skull


    def segment_brain(self, fast=False, only_brain=False): # fast=True to speed up the process, but lessen resolution (1.5mm vs 3mm)
        if __name__ == "__main__":
            input_img = nib.load(self.nii_path)
            if only_brain:
                output_img = totalsegmentator(input_img, fast=fast, roi_subset=["brain"])
            else:
                output_img = totalsegmentator(input_img, fast=fast)
            print("ça marche tu")
            output_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number)
            nib.save(output_img, output_path)
        # Brain is labeled with the number 90
        # Skull is labeled with the number 91
        

    def arteries_mask(self, brain_mask_path=None):
        # brain_mask_path = "nifti/2/totalsegmentator2.nii"  
        brain_mask_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii")
        brain_mask = nib.load(brain_mask_path).get_fdata()
        brain_mask = brain_mask == 90.0
        self.brain = np.where(brain_mask, 1, 0)
        # self.show_3D_array(brain_mask, axis=0) # En y 
        self.arteries = self.brain * self.arteries
        return self.arteries
    
    
    def mask_to_nii(self):
        
        # Nifti file with the head for the registration
        head_img =  nib.load(self.nii_path)
        head_array = head_img.get_fdata()
        head_array = head_array*self.head # image*mask
        head_image = nib.Nifti1Image(head_array, head_img.affine, head_img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "head"+self.file_number)
        nib.save(head_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}")

        # NIfTI file with all the masks
        total_mask = 1*self.head + 2*self.skull + 1*self.arteries
        self.show_3D_array(self.head, axis=2)
        self.show_3D_array(self.skull, axis=2)
        self.show_3D_array(total_mask, axis=2) # En y 
        masked_image = nib.Nifti1Image(total_mask, head_img.affine, head_img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "mask"+self.file_number)
        nib.save(masked_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}")
        



# -----------------------------------------------------------------------------------------------
dicoms_list = ["DICOM_003/Carotid_Angio_0.625mm", "DICOM_010/COW_Angio_0.6_Hv36_3"]

def main(dicoms_list = dicoms_list):
    for i, dicom in enumerate(dicoms_list):
        # print("allo toi", i, dicom)
        ct = Segmentation(dcm_path=dicom, big_output_directory="jspakoi", file_number=i)
        ct.dcm_to_nii() # Trouver comment ne pas avoir besoin de refaire des .nii mais d'avoir le nom nii automatique 
        ct.load_nii()
        ct.apply_threshold()
        ct.keep_largest_island()
        ct.fill_holes()
        ct.remove_arteries()
        # ct.show_3D_array(ct.arteries, axis=0) # En y 

        # Totalsegmentator
        # ct.segment_brain()
        ct.arteries_mask()
        ct.mask_to_nii()
        # ct.show_3D_array(ct.arteries, axis=0) # En y 
        # ct.show_3D_array(ct.arteries, axis=1) # En x 
        # ct.show_3D_array(ct.arteries, axis=2) # En z

# if __name__ == "__main__":
#     main()










class Registration(Segmentation):

    def __init__(self, big_output_directory="processed_files", file_number=0, fixed_img_path='icbm_avg_152_t1_tal_lin.nii'):
        self.big_output_directory = big_output_directory
        self.file_number = str(file_number)
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number)
        self.moving_img_path = os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii") # Complete head without metal frame        
        self.moving_img = ants.image_read(self.moving_img_path, reorient='IAL')
        self.fixed_img = ants.image_read(fixed_img_path, reorient='IAL')

        self.fwd_df_transform = None
        self.fwd_a_transform = None 
        self.inv_a_transform = None
        self.inv_df_transform = None

        self.vox_lpa = np.array([25, 107, 6])
        self.vox_rpa = np.array([25, 107, 173])
        self.vox_nasion = np.array([28, 4, 90])


    def register(self, show=True):
        transformation = ants.registration(fixed=self.fixed_img, moving=self.moving_img, type_of_transform='SyN', verbose=True)
        print("TRANSFORMATION : ", transformation)

        # Sauver les transformations
        import shutil
        shutil.copy(transformation['fwdtransforms'][0], os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz"))
        ants.write_transform(transform=ants.read_transform(transformation['fwdtransforms'][1]), filename=os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat"))
        ants.write_transform(transform=ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])), filename=os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat"))
        shutil.copy(transformation['invtransforms'][1], os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz"))

        self.fwd_df_transform = ants.read_transform(transformation['fwdtransforms'][0]) # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
        self.fwd_a_transform = ants.read_transform(transformation['fwdtransforms'][1]) # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
        self.inv_a_transform = ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])) # .mat -> affine transform (a) 
        self.inv_df_transform = ants.read_transform(transformation['invtransforms'][1]) # .nii.gz -> deformation field (df)
    
        if show:
            self.show_3D_array(arr=self.moving_img.numpy())
            self.show_3D_array(arr=self.fixed_img.numpy())
            self.show_3D_array(arr=transformation['warpedmovout'].numpy()) # moving_image déformée
            self.show_3D_array(arr=transformation['warpedfixout'].numpy()) # fixed_image déformée


    def read_transforms(self):
        self.fwd_df_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz")) # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
        self.fwd_a_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat")) # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
        self.inv_a_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat")) # .mat -> affine transform (a) 
        self.inv_df_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz")) # .nii.gz -> deformation field (df)


    def find_lpa_rpa_nasion(self):

        # LPA : automatically identify on patient
        lpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.vox_lpa)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, lpa_pt_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, lpa_pt_patient_space)
        lpa_vox_patient_space =  ants.transform_physical_point_to_index(self.moving_img, lpa_pt_patient_space)
        print("Point espace patient final :", lpa_pt_patient_space)
        print('Voxel final', lpa_vox_patient_space)

        # RPA : automatically identify on patient
        rpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.vox_rpa)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, rpa_pt_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, rpa_pt_patient_space)
        rpa_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, rpa_pt_patient_space)
        print("Point espace patient final :", rpa_pt_patient_space)
        print('Voxel final', rpa_vox_patient_space)

        # nasion : automatically identify on patient
        nas_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.vox_nasion)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, nas_pt_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, nas_pt_patient_space)
        nas_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, nas_pt_patient_space)
        print("Point espace patient final :", nas_pt_patient_space)
        print('Voxel final', nas_vox_patient_space)



id = Registration(big_output_directory="nifti", file_number=2, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
# id.register()
id.read_transforms()
id.find_lpa_rpa_nasion()
# AJOUTER L'AFFICHAGE DES PTS TROUVÉS





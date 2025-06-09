import dicom2nifti
import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np
import glob


dicoms_list = ["DICOM_003/Carotid_Angio_0.625mm", "DICOM_010/COW_Angio_0.6_Hv36_3"]

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
        
        self.no_arteries_array = None
        self.arteries = None
        self.head = None
        self.skull = None
        self.air = None
  

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


    def segment_brain(self, fast=False, only_brain=False):
        if __name__ == "__main__":
        # def segment(input_path="nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz", output_path="nifti/2/totalsegmentator2", fast=False, only_brain=False):
            input_img = nib.load(self.nii_path)
            if only_brain:
                output_img = totalsegmentator(input_img, fast=fast, roi_subset=["brain"]) # Mettre True pour plus vite
            else:
                output_img = totalsegmentator(input_img, fast=fast) # Mettre True pour plus vite
            print("ça marche tu")
            output_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number)
            nib.save(output_img, output_path)

        # Brain is labeled with the number 90
        # Skull is labeled with the number 91
        
# segment(input_path="nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz", output_path="testtii/2/totalsegmentator2")
# segment(input_path="nifti/1/cropped_301_carotid_angio_0625mm.nii.gz", output_path="nifti/1/totalsegmentator1")


    def arteries_mask(self, brain_mask_path=None):
        # brain_mask_path = "nifti/2/totalsegmentator2.nii"  
        brain_mask_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number)
        brain_mask = nib.load(brain_mask_path).get_fdata()
        brain_mask = brain_mask == 90.0
        brain_mask = np.where(brain_mask, 1, 0)
        # self.show_3D_array(brain_mask, axis=0) # En y 
        self.arteries = brain_mask * self.arteries
        return self.arteries
    
    
    def mask_to_nii(all0 = "jsp"):

        segm_img = nib.load("nifti/2/totalsegmentator2.nii")
        segm_array = segm_img.get_fdata()

        head_img =  nib.load("nifti/2/cropped_6_cow_angio__06__hv36__3.nii.gz")
        head_array = head_img.get_fdata()
        brain = np.where(segm_array == 1, head_array, -1000) # Put -1000 where the mask is 0
        # brain = head_array*segm_array

        plt.imshow(brain[:,:,200], origin="lower", cmap="gist_gray") # y, x, z
        plt.show()

        # Create a new NIfTI image
        brain_image = nib.Nifti1Image(brain, head_img.affine, head_img.header)

        # Save the new NIfTI image under the same path 
        nifti_path = "nifti/2/brain2"

        nib.save(brain_image, nifti_path)
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
        ct.segment_brain()
        # ct.arteries_mask()
        # ct.show_3D_array(ct.arteries, axis=0) # En y 
        # ct.show_3D_array(ct.arteries, axis=1) # En x 
        # ct.show_3D_array(ct.arteries, axis=2) # En z

if __name__ == "__main__":
    main()


# ct = Segmentation()
# print(nib.load("nifti/icbm_avg_152_t1_tal_lin_headmask.nii").header)

# Rajouter la segmentation du cerveau fait par TotalSegmentator (deep learning)
arr = nib.load("nifti/icbm_avg_152_t1_tal_lin_headmask.nii").get_fdata()
arr = nib.load("nifti/icbm_avg_152_t1_tal_lin.nii").get_fdata() #* arr # Marche pas, il faudrait resample
# ct.show_3D_array(arr=arr, axis=0)
# ct.show_3D_array(arr=arr, axis=1)
# ct.show_3D_array(arr=arr, axis=2)


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
        ct.segment_brain()
        ct.arteries_mask()
        ct.mask_to_nii()
        # ct.show_3D_array(ct.arteries, axis=0) # En y 
        # ct.show_3D_array(ct.arteries, axis=1) # En x 
        # ct.show_3D_array(ct.arteries, axis=2) # En z
        # id = Registration(big_output_directory="nifti", file_number=2, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
        # # id.register()
        # id.read_transforms()
        # id.find_registered_lpa_rpa_nasion()
        # # AJOUTER L'AFFICHAGE DES PTS TROUVÉS


# if __name__ == "__main__":
#     main()


# BIZARREEEEEEEE : S'ASSURER QUE Z=AXE0, X=AXE1 ET Y=AXE2 TEL QUE DHABITUDE : PAS LE CAS DANS LE MASQUE CI-HAUT







# ------------------------------------------------------------------------------------------------------------------------------

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

        self.lpa_vox_normal_space = np.array([25, 107, 6])
        self.rpa_vox_normal_space = np.array([25, 107, 173])
        self.nas_vox_normal_space = np.array([28, 4, 90])
        self.lpa_vox_patient_space = None
        self.rpa_vox_patient_space = None
        self.nas_vox_patient_space = None

        self.filled_y_slices = []
        self.head = nib.load(os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii")).get_fdata()
        # Swap axes 0 and 2 and mirrors axis 1 and 2 because the .nii files gives (y, x, z) instead of (z, x, y) otherwise
        # It is for the coordinates to fit with those given by ants
        self.head = np.flip(np.flip(np.transpose(np.where(self.head>=1, 1, 0), (2, 1, 0)), axis=1), axis=2)
        # self.head = np.flip(np.flip(np.flip(np.where(self.head>=1, 1, 0), axis=1), axis=2), axis=0)
        self.registered_nasion = None
        self.nasion = None


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


    def find_registered_lpa_rpa_nasion(self):

        # LPA : automatically identify on patient
        lpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.lpa_vox_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, lpa_pt_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, lpa_pt_patient_space)
        self.lpa_vox_patient_space =  ants.transform_physical_point_to_index(self.moving_img, lpa_pt_patient_space)
        print("Point espace patient final :", lpa_pt_patient_space)
        print('Voxel final', self.lpa_vox_patient_space)

        # RPA : automatically identify on patient
        rpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.rpa_vox_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, rpa_pt_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, rpa_pt_patient_space)
        self.rpa_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, rpa_pt_patient_space)
        print("Point espace patient final :", rpa_pt_patient_space)
        print('Voxel final', self.rpa_vox_patient_space)

        # nasion : automatically identify on patient
        nas_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.nas_vox_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, nas_pt_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, nas_pt_patient_space)
        self.nas_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, nas_pt_patient_space)
        print("Point espace patient final :", nas_pt_patient_space)
        print('Voxel final', self.nas_vox_patient_space)


    def fill_cavities(self):
        from scipy.ndimage import binary_fill_holes

        for i in range(0, len(self.head[1,1,:])):
            # Fill holes in the x-z and y-z plane 
            original = self.head[:,:,i]
            filled = binary_fill_holes(self.head[:,:,i])
            if not np.array_equal(original,filled):
                self.filled_y_slices.append(i)
            self.head[:,:,i] = filled
            self.head[:,i,:] = binary_fill_holes(self.head[:,i,:])
        print(self.filled_y_slices)
        for i in range(0, len(self.head[:,1,1])):
            # Fill holes in the x-y plane
            self.head[i,:,:] = binary_fill_holes(self.head[i,:,:])
        
        return self.head, self.filled_y_slices
    

    def find_nasion(self, window=20):
        nasion_x = 47
        nasion_y = 237
        nasion_z = 96
        self.registered_nasion = nasion_x, nasion_y, nasion_z

        ROI_nas = self.head[nasion_z-window:nasion_z+window,nasion_x-window:nasion_x+window,nasion_y-window:nasion_y+window]
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(ROI_nas, axis = 1) # axis=2 donne somme en y, axis=0 donne somme en z
        plt.imshow(counts_x, origin="lower")
        plt.show()

        # Index of the maximal values for rows and minimal values for columns
        max_index_row = (np.argmin(counts_x, axis=0))
        min_index_column = (np.argmax(counts_x, axis=1))

        optimize_minmax = []
        for column, position_min_in_column in enumerate(min_index_column):
            max_index_in_row_of_min_column = max_index_row[position_min_in_column]
            optimize_minmax.append(np.abs(column-max_index_in_row_of_min_column))
        i = np.argmin(optimize_minmax)
        j = min_index_column[i]
        nasion_row = np.where(counts_x[i,:] == np.max(counts_x[i,:]))[0] 
        nasion_column = np.where(counts_x[:,j] == np.min(counts_x[:,j]))[0]
        nasion_y_final = int(np.mean(nasion_row)) + (nasion_y - window)
        nasion_z_final = int(np.mean(nasion_column)) + (nasion_z - window)
        nasion_x_final = (2*window) - counts_x[int(np.mean(nasion_row)),int(np.mean(nasion_column))] + (nasion_x - window)

        # Nasion en (x, y, z)
        self.nasion = nasion_x_final, nasion_y_final, nasion_z_final
        return self.nasion


    def check_nasion(self):
        # # Si l'axe x passe à travers le nasion et règle de la main droite
        # # Plan axial : Valeur fixe de z
        plt.imshow(self.head[self.nasion[2],:,:], origin="lower")
        plt.scatter([self.nasion[1]], [self.nasion[0]], c="r")
        plt.scatter([self.registered_nasion[1]], [self.registered_nasion[0]], c="b")
        plt.show()
        # # Plan coronal : Valeur fixe de x
        plt.imshow(self.head[:,self.nasion[0],:], origin="lower")
        plt.scatter([self.nasion[1]], [self.nasion[2]], c="r")
        plt.scatter([self.registered_nasion[1]], [self.registered_nasion[2]], c="b")
        plt.show()
        # # Plan sagittal : Valeur fixe de y
        plt.imshow(self.head[:,:,self.nasion[1]], origin="lower")
        plt.scatter([self.nasion[0]], [self.nasion[2]], c="r")
        plt.scatter([self.registered_nasion[0]], [self.registered_nasion[2]], c="b")
        plt.show()


    def find_lpa_rpa(self, window=50):
        lpa_x = 223
        lpa_y = 99#84
        lpa_z = 63
        self.registered_lpa = lpa_x, lpa_y, lpa_z

        if lpa_y in self.filled_y_slices:
            print("allo test passé")
            self.filled_y_slices = np.array(self.filled_y_slices)
            print(self.filled_y_slices)   
            index_lpa_y = np.where(self.filled_y_slices == lpa_y)[0][0]
            print(index_lpa_y)
            for i in range(1, index_lpa_y):
                if self.filled_y_slices[index_lpa_y-i] != lpa_y-i:
                    return self.filled_y_slices[index_lpa_y-i+1]-1
        else:
            index_lpa = np.argmin(np.abs(lpa_y-self.filled_y_slices))
            return self.filled_y_slices[index_lpa]-1
                
            # mettre que si self.filled_y_slices[index-i] != 87-i
            # return self.filled_y_slices[index-i]
            # else: return self.filled_y_slices[0]
        # ROI_lpa = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y-window:lpa_y+window]
        # self.show_3D_array(ROI_lpa, axis=2) # y axis





id = Registration(big_output_directory="jspakoi", file_number=0, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
# id.register()
# id.read_transforms()
# id.find_registered_lpa_rpa_nasion()


# id.show_3D_array(id.head, axis=2)
id.fill_cavities()
# id.show_3D_array(id.head, axis=2)
# id.find_nasion()
# id.check_nasion()
print(id.find_lpa_rpa())

id.show_3D_array(id.head)
id.read_transforms()
id.find_registered_lpa_rpa_nasion()

# AJOUTER L'AFFICHAGE DES PTS TROUVÉS





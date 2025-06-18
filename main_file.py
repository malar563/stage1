import dicom2nifti
import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np
import ants
import time


class Segmentation:
    """Processing DICOM files, segmenting the head and storing various image masks."""
    
    def __init__(self, dcm_path="DICOM_010/COW_Angio_0.6_Hv36_3", big_output_directory="processed_files", file_number=0):
        """
        This constructor receive the path of the DICOM file to be processed,
        sets up file paths for NIfTI outputs and initializes storage for multiple segmentation masks.

        Arguments
        ---------
        dcm_path : string
            Path to the directory containing the DICOM files to be processed.
        
        big_output_directory : string
            Base directory where processed NIfTI files will be saved.
        
        file_number : int
            Integer identifier for the file being processed, used to create subdirectories.

        Attributes
        ----------
        dcm_path : string
            Path to the DICOM directory.

        nii_path : string or None
            Full path to the primary NIfTI file after conversion.

        array : numpy.ndarray
            Placeholder for the primary loaded DICOM or NIfTI image data.

        resolution : tuple or None
            Resolution information extracted from the image metadata.

        big_output_directory : string
            Path to the top-level directory for storing outputs.

        file_number : string
            String version of the file number used in folder naming.

        nifti_output_directory : string
            Full path to the directory where NIfTI outputs will be stored.

        Future attributes
        -----------------
        Masks you can access at one point in the segmentation process :
            -> self.head, self.skull, self.no_arteries_array, self.arteries
            -> self.brain_totalsegmentator and self.skull_totalsegmentator            

        Notes
        -----
        IMPORTANT :
            In this class, when displaying an array, note that the index order is
            (Y, X, Z)    
        """

        self.dcm_path = dcm_path # Path to the DICOM directory.
        self.big_output_directory = big_output_directory # Path to the top-level directory for storing outputs.
        self.file_number = str(file_number) # String version of the file number used in folder naming.
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number) # Full path to the directory where NIfTI outputs will be stored.
        
        # Checking if the DICOM has already been converted. If not, converting it.
        try:
            self.nii_path = [f for f in os.listdir(self.nifti_output_directory) if f.startswith('cropped')][0] 
            self.nii_path = os.path.join(self.nifti_output_directory, self.nii_path) # Full path to the primary NIfTI file after conversion.
            self.img = nib.load(self.nii_path)
            self.array = img.get_fdata() # Placeholder for loaded NIfTI image data (primary NIfTI file)
            self.resolution = np.abs(img.affine[0][0]), np.abs(img.affine[1][1]), np.abs(img.affine[2][2]) # Just in case you want to see
            print(f"NIfTI found : {self.nii_path}")
        except:
            print("No processed NIfTI file in the directory. Processing the specified DICOM file...")
            self.dcm_to_nii()
            img = nib.load(self.nii_path)
            self.array = img.get_fdata() # Placeholder for loaded NIfTI image data (primary NIfTI file)
            self.resolution = np.abs(img.affine[0][0]), np.abs(img.affine[1][1]), np.abs(img.affine[2][2]) # Just in case you want to see

    def dcm_to_nii(self, crop="yes"):
        """
        Convert a DICOM series to a NIfTI file and optionally crop it along the z-axis.

        Uses `dicom2nifti` to convert DICOM files in `self.dcm_path` to a `.nii.gz` file 
        saved in `self.nifti_output_directory`. If `crop` is set, the image is cropped 
        based on slice thickness (last 256 or 512 slices kept) and saved with a new name.

        Arguments
        ----------
        crop : str or None
            If not None, crop the volume depending on pixel spacing.

        Sets
        ----    
        self.nii_path : string
            Full path to the primary NIfTI file after conversion.    

        Files Created
        -------------
        - {automatically_named_file}.nii.gz : Equivalent of the DICOM folder converted
        - cropped_{automatically_named_file}.nii.gz : Cropped according to the resolution file
        """

        # Create the output_directory file
        os.makedirs(self.nifti_output_directory, exist_ok=True)

        # Convert DICOM to NIfTI (compression=False -> .nii instead of .nii.gz)
        dicom2nifti.convert_directory(self.dcm_path, self.nifti_output_directory, compression=True)

        # Find the generated file in the output folder
        nifti_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith('.nii.gz')]
        nifti_path = os.path.join(self.nifti_output_directory, nifti_files[0]) # Use the first .nii.gz file found
        print(f"NIfTI generated : {nifti_path}")

        if crop is not None:
            # Load the image with nibabel
            nifti_image = nib.load(nifti_path)

            # Crop the image depending on the resolution 
            header = cropped_image.header
            pix_dim, pix_z = header["pixdim"][1:4], header["pixdim"][3]
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

            # No other purpose than seeing it
            shape = cropped_image.shape
            affine = cropped_image.affine
            data = cropped_image.get_fdata()
            # print("Pixel dimensions :", pix_dim, pix_z)
            # print("Dimensions :", shape)
            # print("Entête :", header)
            # print("Affine :", affine)
            # print("data :", data)

            self.nii_path = nifti_path

   

    def show_3D_array(self, arr, axis=0, pt=None): # y=0, x=1, z=2
        """
        Display a 3D array slice-by-slice using a matplotlib slider.

        Arguments
        ----------
        arr : numpy.ndarray
            3D array to visualize.

        axis : int
            Axis along which to slice (0=y, 1=x, 2=z).

        pt : tuple or None
            Optional (x, y) coordinates to highlight on each slice.

        Notes
        -----
        Opens an interactive window to scroll through slices of the array.
        """
        from matplotlib.widgets import Slider

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # Initial slice index
        index = arr.shape[axis] // 2
        if axis == 0:
            img = ax.imshow(arr[index, :, :], cmap="gray", origin="lower")
            if pt is not None:
                ax.scatter([pt[0]],[pt[1]], c="r")
        elif axis == 1:
            img = ax.imshow(arr[:, index, :], cmap="gray", origin="lower")
            if pt is not None:
                ax.scatter([pt[0]],[pt[1]], c="r")
        else:
            img = ax.imshow(arr[:, :, index], cmap="gray", origin="lower")
            if pt is not None:
                ax.scatter([pt[0]],[pt[1]], c="r")

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
        """
        Apply intensity thresholds to segment different anatomical structures.

        Generates binary masks for head, air, skull, arteries, and a no-arteries region 
        based on fixed intensity thresholds applied to `self.array`.

        Arguments
        ----------
        threshold_head : int
            Lower bound for head tissue detection.
        
        threshold_skull : int
            Lower bound for skull segmentation.
        
        threshold_no_arteries : int
            Threshold above which regions are considered free of arteries.
        
        threshold_arteries : int
            Lower bound for artery detection.

        Sets
        ----
        self.head : numpy.ndarray
        self.skull : numpy.ndarray
        self.no_arteries_array : numpy.ndarray
        self.arteries : numpy.ndarray
        """
        # Array with "True" where it is, and "False" where it is not
        thresholded_head = self.array >= threshold_head
        thresholded_skull = self.array >= threshold_skull
        thresholded_no_arteries = self.array >= threshold_no_arteries
        thresholded_arteries = self.array >= threshold_arteries
        # Put the value 1 if True, and 0 if False
        self.head = np.where(thresholded_head, 1, 0)
        self.skull = np.where(thresholded_skull, 1, 0)
        self.no_arteries_array = np.where(thresholded_no_arteries, 1, 0)
        self.arteries = np.where(thresholded_arteries, 1, 0)
    
    
    def keep_largest_island(self):
        """
        Keep only the largest connected component in each binary mask.

        Applies 3D connected component labeling to `head`, `skull` and `no_arteries_array`,
        and retains only the largest region in each.

        Updates
        -------
        self.head : numpy.ndarray
            Binary mask of the where only the biggest component (head) is left.
        self.skull : numpy.ndarray
        self.no_arteries_array : numpy.ndarray
        """
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
    

    def fill_holes(self):
        """
        Fill internal holes in the skull mask.
        Uses binary morphology to fill enclosed voids in `self.skull`.

        Updates
        -------
        self.skull : numpy.ndarray
            Binary mask of skull without hole.
        """
        from scipy.ndimage import binary_fill_holes
        self.skull = binary_fill_holes(self.skull)



    def remove_arteries(self, max_distance = 3): 
        # max_distance = 3 with skull threshold = 200 and no_arteries threshold = 500 works fine, but not totally systematic
        """
        Remove arteries from the skull mask based on proximity to artery-free regions.

        Uses a distance transform on `no_arteries_array` to exclude skull voxels 
        too far from artery-free regions, then slightly dilates the result.

        Arguments
        ----------
        max_distance : int
            Maximum distance (in voxels) to keep skull regions near artery-free zones.

        Updates
        -------
        self.skull : numpy.ndarray
            Binary mask of skull where arteries are removed.
        """
        from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure

        self.no_arteries_array = self.no_arteries_array != 1
        distance = distance_transform_edt(self.no_arteries_array) # Compute distance of the voxel from closet 0 value
        close_to_bone = distance < max_distance 
        self.skull = self.skull & close_to_bone # Voxels in the skull mask must be in the original mask AND close to higher HU value bones
        self.skull = binary_dilation(self.skull, generate_binary_structure(3, 1))


    def segment_brain(self, fast=False, only_brain=False):
        """
        Run TotalSegmentator to segment the head (brain and skull mainly).

        Loads the NIfTI image from `self.nii_path` and performs segmentation using 
        TotalSegmentator. Can restrict to brain only or segment the full head.

        Arguments
        ----------
        fast : bool
            If True, uses lower resolution (3mm instead of 1.5mm) for faster segmentation. 
            (not recommanded for our purposes)

        only_brain : bool
            If True, segments only the brain (label 90); not recommanded, the skull is needed later in the process.
            Otherwise, includes the skull (label 91) and more.

        File Created
        -------------
        - totalsegmentator{file_number}.nii : Segmented head     

        Notes
        -----
        Only runs when the script is executed directly.
        IMPORTANT : 
            Brain is labeled with the number 90
            Skull is labeled with the number 91
        """
        if __name__ == "__main__":
            input_img = nib.load(self.nii_path)
            if only_brain:
                output_img = totalsegmentator(input_img, fast=fast, roi_subset=["brain"])
            else:
                output_img = totalsegmentator(input_img, fast=fast)
            print("Segmentation with TotalSegmentator has been completed")
            output_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number)
            nib.save(output_img, output_path)


    def arteries_and_totalsegmentator_mask(self):
        """
        Combine TotalSegmentator brain mask with existing artery mask.

        Loads TotalSegmentator output, extracts brain (label 90) and skull (label 91) masks, 
        and updates `self.arteries` to keep only arteries within the brain.

        Sets
        -----
        self.brain_totalsegmentator : numpy.ndarray
            Binary mask of the brain obtained with TotalSegmentator. 
        self.skull_totalsegmentator : numpy.ndarray
            Binary mask of the skull obtained with TotalSegmentator.
        
        Updates
        -------
        self.arteries : numpy.ndarray
            Binary mask of brain arteries.
        """
        totalsegmentator_mask = nib.load(os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii")).get_fdata()
        self.brain_totalsegmentator = np.where(totalsegmentator_mask == 90, 1, 0)
        self.skull_totalsegmentator = np.where(totalsegmentator_mask == 91, 1, 0) 
        self.arteries = self.brain_totalsegmentator * self.arteries

    

    
    def mask_to_nii(self, iter_erosion=3):
        """
        Save head and masks as NIfTI files, with skull refinement.

        Creates two NIfTI files:
        - One containing the original image masked by the head (useful for registration).
        - One combining the air (label=0), head (label=1), arteries (label=2) and skull (label>=3),
          and artery masks with skull refinement using erosion and soft tissue overlap.

        Arguments
        ----------
        iter_erosion : int
            Number of binary erosion iterations to refine the skull mask.

        Files Created
        -------------
        - head{file_number}.nii : Non masked head image. 
        - mask{file_number}.nii : Labeled mask
        """
        # Nifti file with the HU units of the whole head for the registration
        head_img =  nib.load(self.nii_path)
        head_array = head_img.get_fdata() # primary nifti file data
        head_array = np.where(self.head == 1, head_array, -1000) # HU units where the mask is 1, -1000 where the mask is 0
        head_image = nib.Nifti1Image(head_array, head_img.affine, head_img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "head"+self.file_number)
        nib.save(head_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}.nii")

        # NIfTI file with all the masks
        total_mask = 1*self.head + 2*self.skull + 1*self.arteries
        # Improve the mask of the skull
        from scipy.ndimage import binary_erosion, generate_binary_structure
        mask_soft_tissues = np.where(total_mask == 1, 1, 0)
        mask_eroded_skull = binary_erosion(self.skull_totalsegmentator, structure=generate_binary_structure(3,1), iterations=iter_erosion)
        not_included_skull = 3*mask_soft_tissues*mask_eroded_skull
        self.skull = self.skull + (mask_soft_tissues*mask_eroded_skull)
        total_mask = total_mask + not_included_skull

        masked_image = nib.Nifti1Image(total_mask, head_img.affine, head_img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "mask"+self.file_number)
        nib.save(masked_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}.nii")


 



# -----------------------------------------------------------------------------------------------
dicoms_list = ["DICOM_003/Carotid_Angio_0.625mm", "DICOM_010/COW_Angio_0.6_Hv36_3"]


# CHECKER LES AXES PARTOUT POUR ÊTRE SÛR QUE C'EST CHILL
def main(dicoms_list = dicoms_list):
    for i, dicom in enumerate(dicoms_list):
        start = time.time()
        ct = Segmentation(dcm_path=dicom, big_output_directory="jspakoi", file_number=i) 
        ct.apply_threshold()
        ct.keep_largest_island()
        ct.fill_holes()
        ct.remove_arteries()

        # Totalsegmentator
        # ct.segment_brain()
        ct.arteries_and_totalsegmentator_mask()
        ct.mask_to_nii()
        ct.show_3D_array(ct.skull, axis=2) # En z

        # id = Registration(big_output_directory="nifti", file_number=2, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
        # # id.register()
        # id.read_transforms()
        # id.find_registered_lpa_rpa_nasion()
        # # AJOUTER L'AFFICHAGE DES PTS TROUVÉS
        print(f"Time to segment file {ct.nii_path} : {time.time() - start} seconds")


# if __name__ == "__main__":
#     main()


# BIZARREEEEEEEE : S'ASSURER QUE Z=AXE0, X=AXE1 ET Y=AXE2 TEL QUE DHABITUDE : PAS LE CAS DANS LE MASQUE CI-HAUT







# ------------------------------------------------------------------------------------------------------------------------------

class Registration(Segmentation):

    def __init__(self, big_output_directory="processed_files", file_number=0, fixed_img_path='icbm_avg_152_t1_tal_lin.nii'):
        """
        Initialize the Registration class for image alignment and landmark localization.

        Loads moving (patient) and fixed (atlas) images, prepares file paths, loads head/mask data, 
        and precomputes resolution and flipped/transposed arrays for visualization or processing.

        Arguments
        ----------
        big_output_directory : str
            Directory where processed folder of files are stored.
        file_number : int
            Identifier used to locate the directory, load specific NIfTI and mask files.
        fixed_img_path : str
            Path to the fixed atlas image used for registration.

        Attributes
        ----------
        moving_img : ants.ANTsImage
            Patient image to be registered.
        fixed_img : ants.ANTsImage
            Atlas image used as reference.
        lpa_vox_normal_space, rpa_vox_normal_space, nas_vox_normal_space : np.ndarray
            Landmark coordinates in the normalized space (they were determined visually on the fixed_img).
        head, filled_head : np.ndarray
            Processed head mask arrays, flipped and transposed for display.
        resolution : tuple of float
            Spatial resolution (voxel spacing) in mm along each axis.

        Future attributes
        -----------------
        Attributes you can access at one point in the identification process :
            -> fwd_df_transform, fwd_a_transform, inv_a_transform, inv_df_transform 
            -> registered_nasion, registered_lpa, registered_rpa, nasion, lpa, rpa :
            -> arteries

        Notes
        -----
        IMPORTANT :
            In this class, when displaying an array, note that the index order is
            (Z, X, Y) and NOT (Y, X, Z) as it was in the Segmentation class  
        """
        
        self.big_output_directory = big_output_directory
        self.file_number = str(file_number)
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number)
        self.moving_img_path = os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii")
        self.fixed_img_path = fixed_img_path

        self.moving_img = ants.image_read(self.moving_img_path, reorient='IAL') # Image to be registered
        self.fixed_img = ants.image_read(fixed_img_path, reorient='IAL') # Reference image for registration

        # Landmark coordinates determined visually
        self.lpa_vox_normal_space = np.array([25, 107, 6])
        self.rpa_vox_normal_space = np.array([25, 107, 173])
        self.nas_vox_normal_space = np.array([28, 4, 90])

        self.filled_y_slices = [] # Used to determine positions (lpa and rpa)
        mask_img = nib.load(os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii"))
        self.head = mask_img.get_fdata()
        # IMPORTANT : flipping the images for better display. 
        # From now on, it is [z,x,y] and not [y,x,z] as it was in the Segmentation class
        self.filled_head = np.flip(np.flip(np.transpose(np.where(self.head>=1, 1, 0), (2, 1, 0)), axis=1), axis=2)
        self.head = np.flip(np.flip(np.transpose(np.where(self.head>=1, 1, 0), (2, 1, 0)), axis=1), axis=2)

        self.lpa=(0,0,2)
        self.rpa=(0,0,9) 


    def register(self, show=False):
        """
        Perform non-linear registration (SyN) between the moving (patient) and fixed (atlas) images.

        Uses ANTsPy to compute affine and deformable transformations. Saves the transformation 
        files (forward and inverse) in the output directory, and stores them as class attributes 
        for later use (e.g., point or image transformation).

        Parameters
        ----------
        show : bool, optional
            If True, displays the moving image, fixed image, and the results of the registration 
            (warped moving and fixed outputs) using the show_3D_array method.

        Sets
        ----
        self.fwd_df_transform : ants.ANTsTransform
            Forward deformation field (.nii.gz).
        self.fwd_a_transform : ants.ANTsTransform
            Forward affine transform (.mat).
        self.inv_a_transform : ants.ANTsTransform
            Inverted affine transform (.mat).
        self.inv_df_transform : ants.ANTsTransform
            Inverse deformation field (.nii.gz).

        Files Created
        -------------
        - fwd{file_number}.nii.gz  : Forward deformation field.
        - fwd{file_number}.mat     : Forward affine transform.
        - inv{file_number}.mat     : Inverted affine transform.
        - inv{file_number}.nii.gz  : Inverse deformation field.
        """

        # Compute the optimal forward and inverse transforms
        transformation = ants.registration(fixed=self.fixed_img, moving=self.moving_img, type_of_transform='SyN', verbose=True)
        print("TRANSFORMATION : ", transformation)

        # Save the transforms in the working directory
        import shutil
        shutil.copy(transformation['fwdtransforms'][0], os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz"))
        ants.write_transform(transform=ants.read_transform(transformation['fwdtransforms'][1]), filename=os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat"))
        ants.write_transform(transform=ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])), filename=os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat"))
        shutil.copy(transformation['invtransforms'][1], os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz"))

        # Associate it to a variable for later use
        self.fwd_df_transform = ants.read_transform(transformation['fwdtransforms'][0]) # .nii.gz -> deformation field (df) : fwd_df_transform != inv_df_transform
        self.fwd_a_transform = ants.read_transform(transformation['fwdtransforms'][1]) # .mat -> affine transform (a) : fwd_a_transform = inv_a_transform 
        self.inv_a_transform = ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])) # .mat -> affine transform (a) 
        self.inv_df_transform = ants.read_transform(transformation['invtransforms'][1]) # .nii.gz -> deformation field (df)
    
        if show:
            self.show_3D_array(self.moving_img.numpy()) # moving_img (before reg.)
            self.show_3D_array(self.fixed_img.numpy()) # fixed_img (before reg.)
            self.show_3D_array(transformation['warpedmovout'].numpy()) # Warped moving_img (after reg.)
            self.show_3D_array(transformation['warpedfixout'].numpy()) # Warped fixed_img (after reg.)


    def read_transforms(self):
        """
        Load precomputed ANTs transformation files (affine and deformation field) 
        from the output directory and store them as class attributes.

        This method is useful when transformations were previously saved to disk 
        (e.g., after a registration step) and need to be reused without re-registering.

        Sets
        ----
        self.fwd_df_transform : ants.ANTsTransform
            Forward deformation field (.nii.gz).
        self.fwd_a_transform : ants.ANTsTransform
            Forward affine transform (.mat).
        self.inv_a_transform : ants.ANTsTransform
            Inverse affine transform (.mat).
        self.inv_df_transform : ants.ANTsTransform
            Inverse deformation field (.nii.gz).
        """
        self.fwd_df_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz")) # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
        self.fwd_a_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat")) # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
        self.inv_a_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat")) # .mat -> affine transform (a) 
        self.inv_df_transform = ants.read_transform(os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz")) # .nii.gz -> deformation field (df)


    def find_registered_lpa_rpa_nasion(self):
        """
        Compute the patient-space coordinates of anatomical landmarks (LPA, RPA, nasion)
        by applying forward transformations to standard-space (MNI) coordinates.

        Uses the fixed image (atlas) and moving image (patient) to convert known MNI voxel
        positions to patient voxel space through deformation and affine transforms.

        Sets
        -------
        self.registered_lpa : tuple[int, int, int]
            LPA voxel coordinates in patient space (z, x, y).
        self.registered_rpa : tuple[int, int, int]
            RPA voxel coordinates in patient space (z, x, y).
        self.registered_nasion : tuple[int, int, int]
            Nasion voxel coordinates in patient space (z, x, y).

        Notes
        -----
        - Uses `ants.transform_index_to_physical_point()` to get physical coordinates in the atlas.
        - Applies both deformation and affine forward transforms in sequence.
        - Converts transformed physical points back to voxel indices using the patient image.
        - Coordinates are rounded to integer values (z, x, y) to get the position in the array.
        """
        # LPA : automatically identify on patient
        lpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.lpa_vox_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, lpa_pt_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, lpa_pt_patient_space)
        lpa_vox_patient_space =  ants.transform_physical_point_to_index(self.moving_img, lpa_pt_patient_space)
        self.registered_lpa = round(lpa_vox_patient_space[0]), round(lpa_vox_patient_space[1]), round(lpa_vox_patient_space[2])
        print('Voxel final', self.registered_lpa) # (z,x,y)

        # RPA : automatically identify on patient
        rpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.rpa_vox_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, rpa_pt_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, rpa_pt_patient_space)
        rpa_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, rpa_pt_patient_space)
        self.registered_rpa = round(rpa_vox_patient_space[0]), round(rpa_vox_patient_space[1]), round(rpa_vox_patient_space[2])
        print('Voxel final', self.registered_rpa) # (z,x,y)

        # nasion : automatically identify on patient
        nas_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.nas_vox_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, nas_pt_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, nas_pt_patient_space)
        nas_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, nas_pt_patient_space)
        self.registered_nasion = round(nas_vox_patient_space[0]), round(nas_vox_patient_space[1]), round(nas_vox_patient_space[2])
        print('Voxel final', self.registered_nasion) # (z,x,y)


    def fill_cavities(self):
        """
        Fill internal cavities in the binary head mask across all anatomical planes.

        This method processes the mask slice-by-slice along the three orthogonal axes
        (x-y, x-z, y-z) to fill holes within the `self.filled_head` volume.

        Updates
        -------
        self.filled_head : np.ndarray
            Modified binary head mask with filled cavities.
        self.filled_y_slices : list[int]
            Indices of y-slices where cavities were filled in the x-z plane.

        Notes
        -----
        Assumes that the image has equal dimensions in x and y
        """        
        from scipy.ndimage import binary_fill_holes

        for i in range(0, len(self.filled_head[1,1,:])):
            # Fill holes in the x-z and y-z plane 
            original = self.filled_head[:,:,i]
            filled = binary_fill_holes(self.filled_head[:,:,i])
            if not np.array_equal(original,filled): # Check if the slice has been filled
                self.filled_y_slices.append(i)
            self.filled_head[:,:,i] = filled # Assumes equal dimensions in x and y
            self.filled_head[:,i,:] = binary_fill_holes(self.filled_head[:,i,:]) 
        print(self.filled_y_slices)
        for i in range(0, len(self.filled_head[:,1,1])):
            # Fill holes in the x-y plane
            self.filled_head[i,:,:] = binary_fill_holes(self.filled_head[i,:,:])
    

    def find_nasion(self, window=20):
        """
        Refine the position of the nasion using local anatomical data in the filled head mask.

        This function takes an initial estimation of the nasion (from registration) and looks
        within a cubic region of interest (ROI) centered on this point. It analyzes the 
        density of the binary mask across the x-axis to find a more anatomically accurate location.

        Argument
        ----------
        window : int, optional
            Half-width of the cubic ROI (region of interest) centered on the initial nasion estimate
            (default is 20).

        Sets
        -------
        self.nasion : tuple[int, int, int]
            Refined voxel coordinates of the nasion (z, x, y).
        """
        reg_nas_z, reg_nas_x, reg_nas_y = self.registered_nasion

        ROI_nas = self.filled_head[reg_nas_z-window:reg_nas_z+window,
                                   reg_nas_x-window:reg_nas_x+window,
                                   reg_nas_y-window:reg_nas_y+window]
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(ROI_nas, axis = 1) # axis=2 sums in y, axis=0 sums in z
        # plt.imshow(counts_x, origin="lower")
        # plt.show()

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
        nasion_y_final = int(np.mean(nasion_row)) + (reg_nas_y - window)
        nasion_z_final = int(np.mean(nasion_column)) + (reg_nas_z - window)
        nasion_x_final = (2*window) - counts_x[int(np.mean(nasion_row)),int(np.mean(nasion_column))] + (reg_nas_x - window)

        self.nasion = nasion_z_final, nasion_x_final, nasion_y_final # (z, x, y)


    def check_nasion(self):
        """
        Visually validate the refined nasion position against the initially registered one.

        This function plots axial, coronal, and sagittal views of the head mask, overlaying
        both the refined nasion (in red) and the registered nasion (in blue) to assess the
        accuracy of the correction.
        """
        # # Si l'axe x passe à travers le nasion et règle de la main droite
        # # Axial view : fixed z-value
        plt.imshow(self.head[self.nasion[0],:,:], origin="lower")
        plt.scatter([self.nasion[2]], [self.nasion[1]], c="r")
        plt.scatter([self.registered_nasion[2]], [self.registered_nasion[1]], c="b")
        plt.show()
        # # Coronal view : fixed x-value
        plt.imshow(self.head[:,self.nasion[1],:], origin="lower")
        plt.scatter([self.nasion[2]], [self.nasion[0]], c="r")
        plt.scatter([self.registered_nasion[2]], [self.registered_nasion[0]], c="b")
        plt.show()
        # # Sagittal view : fixed y-value
        plt.imshow(self.head[:,:,self.nasion[2]], origin="lower")
        plt.scatter([self.nasion[1]], [self.nasion[0]], c="r")
        plt.scatter([self.registered_nasion[1]], [self.registered_nasion[0]], c="b")
        plt.show()


    def __find_lpa_y__(self, lpa_y):
        self.filled_y_slices = np.array(self.filled_y_slices)

        if lpa_y in self.filled_y_slices:
            print(self.filled_y_slices)   
            index_lpa_y = np.where(self.filled_y_slices == lpa_y)[0][0]
            print(index_lpa_y)
            for i in range(1, index_lpa_y):
                if self.filled_y_slices[index_lpa_y-i] != lpa_y-i:
                    return self.filled_y_slices[index_lpa_y-i+1]-1 # -1 to get the surface of the head
            return self.filled_y_slices[0] - 1 # -1 to get the surface of the head
        else:
            index_lpa = np.argmin(np.abs(self.filled_y_slices-lpa_y))
            return self.filled_y_slices[index_lpa]-1 # -1 to get the surface of the head
        
    def __find_rpa_y__(self, rpa_y):
        self.filled_y_slices = np.array(self.filled_y_slices)

        if rpa_y in self.filled_y_slices:
            print(self.filled_y_slices)   
            index_rpa_y = np.where(self.filled_y_slices == rpa_y)[0][0]
            print(index_rpa_y)
            for i in range(1, len(self.filled_y_slices)-index_rpa_y):
                if self.filled_y_slices[index_rpa_y+i] != rpa_y+i:
                    return self.filled_y_slices[index_rpa_y+i-1]+1 # -1 to get the surface of the head
            return self.filled_y_slices[-1] + 1 # -1 to get the surface of the head
        else:
            index_lpa = np.argmin(np.abs(self.filled_y_slices-rpa_y))
            return self.filled_y_slices[index_lpa]+1 # -1 to get the surface of the head
        
    def find_rpa(self, window=60):
        rpa_x = 228
        rpa_y = 399
        rpa_z = 69
        # rpa_x = 251 # 0625mm
        # rpa_y = 388
        # rpa_z = 135
        rpa_y_final = self.__find_rpa_y__(rpa_y=rpa_y)
        self.registered_rpa = rpa_x, rpa_y_final, rpa_z
        print(self.registered_rpa)
    


    def find_lpa(self, window=60):
        lpa_x = 223
        lpa_y = 84
        lpa_z = 63
        # lpa_x = 239
        # lpa_y = 97
        # lpa_z = 131
        self.registered_lpa = lpa_x, lpa_y, lpa_z

        # Ya absolument rien qui marche ci-bas donc inutile
        lpa_y_final = self.__find_lpa_y__(lpa_y=lpa_y)
        print(lpa_y_final)
        ROI_lpa = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y_final-window:lpa_y_final+window]
        # frame_before_filling = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y_final]
        # frame_start_filling = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y_final+1]
        frame_before_filling = self.head[:,:,lpa_y_final]
        frame_start_filling = self.head[:,:,lpa_y_final+1]
        intersection = ~frame_before_filling & frame_start_filling
        plt.imshow(intersection, origin="lower")
        plt.scatter([lpa_x], [lpa_z], c="r")
        plt.show()

        
        self.show_3D_array(ROI_lpa, axis=2)
        counts_x = np.sum(ROI_lpa, axis=2) # axis=2 donne somme en y, axis=0 donne somme en z
        plt.imshow(counts_x, origin="lower")
        plt.show()


    def save_pts_to_txt(self):
        """
        Save anatomical point coordinates (Nasion, LPA, RPA) in voxel space to a text file.

        File Created
        ------------
        - points{self.file_number}.txt : Text file containing nasion, LPA, and RPA coordinates.

        Notes
        -----
        - The file is written with voxel coordinates reordered as (X, Y, Z) for readability.
        - If the file already exists, it is overwritten.
        """
        txt_path = os.path.join(self.nifti_output_directory, f"points{self.file_number}.txt")
        f = open(txt_path, "w")
        f.write("Point coordinates are given in voxels : (X, Y, Z)\n" \
                f"Nasion : ({self.nasion[1]}, {self.nasion[2]}, {self.nasion[0]})\n" \
                f"LPA : ({self.lpa[1]}, {self.lpa[2]}, {self.lpa[0]})\n" \
                f"RPA : ({self.rpa[1]}, {self.rpa[2]}, {self.rpa[0]})" )


    
    def mca_arteries_mask(self, arterial_territories_path="mni_vascular_territories.nii.gz"):
        """
        Identify and map the Middle Cerebral Artery (MCA) territories from MNI space to patient space.

        This method extracts the MCA regions (labels 4 and 14) from a vascular atlas (by default
        `mni_vascular_territories.nii.gz`), resamples them to the reference space (using `self.fixed_img_path`),
        and maps the patient's segmented arteries to MNI space. It then isolates voxels included in the
        MCA region and the registered arterial segmentation, and maps the result back to patient space.

        Parameters
        ----------
        arterial_territories_path : str, optional
            Path to the arterial territories in MNI space. 
            Default is "mni_vascular_territories.nii.gz".

        Sets
        ----
        self.arteries : ants.ANTsImage
            Binary mask of segmented MCA arteries in patient space.
        """
        # Keeping the MCA territories
        arterial_territories = ants.image_read(arterial_territories_path, reorient='IAL')
        mca_territories = np.where(arterial_territories.numpy() == 4.0, 1, 0) + np.where(arterial_territories.numpy() == 14.0, 1, 0)
        mca_territories = arterial_territories.new_image_like(mca_territories) # copies image information and just changes the data
        
        # Resample to target image (the atlas is supposed to be in the same space than the normalized MRI)
        mri = ants.image_read(self.fixed_img_path, reorient='IAL')
        resampled_mca_territories = ants.resample_image_to_target(mca_territories, mri, verbose=True)
        # # Seeing the superposition between the 2 images
        # superposition = (arterial_territories.numpy()[:-1,:-1,:-1]/np.max(arterial_territories.numpy()))+(irm.numpy()/np.max(irm.numpy()))
        # self.show_3D_array(superposition)

        # Getting the arteries mask
        self.arteries = ants.image_read(os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii"), reorient="IAL")
        arteries_only = np.where(self.arteries.numpy()==2,1,0)
        # arteries_only = self.arteries.numpy() # Pour mieux voir, mais ne sera pas dans la version finale
        self.arteries = self.arteries.new_image_like(arteries_only)
        self.show_3D_array(self.arteries.numpy())

        # Bringing the arteries mask in the normalized space 
        registered_arteries = ants.apply_transforms(fixed=resampled_mca_territories, moving=self.arteries, transformlist=[os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat"), os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz")])
        self.show_3D_array(registered_arteries.numpy())
        # Keeping the arteries of the mask included in the MC territories
        normalized_mca_arteries = registered_arteries.numpy()*resampled_mca_territories.numpy()
        self.show_3D_array(normalized_mca_arteries)

        # Bringing the MCA arteries back to the patient's space
        normalized_mca_arteries = registered_arteries.new_image_like(normalized_mca_arteries)
        patient_mca_arteries = ants.apply_transforms(fixed=self.arteries, moving=normalized_mca_arteries, transformlist=[os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat"), os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz")])
        
        self.show_3D_array(patient_mca_arteries.numpy())
        counts_z = np.sum(patient_mca_arteries.numpy(), axis = 0) # axis=2 donne somme en y, axis=0 donne somme en z
        plt.imshow(counts_z, origin="lower")
        plt.show()


                





id = Registration(big_output_directory="jspakoi", file_number=0, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')
# id.register()

id.read_transforms()
id.find_registered_lpa_rpa_nasion()

# id.show_3D_array(nib.load('icbm_avg_152_t1_tal_lin.nii').get_fdata(), axis=0)
# id.show_3D_array(nib.load('jspakoi/0/head0.nii').get_fdata(), axis=0)
# id.show_3D_array(id.head, axis=0)
# id.show_3D_array(id.moving_img.numpy(), axis=0)
# id.show_3D_array(id.fixed_img.numpy(), axis=0)

# id.mca_arteries_mask()
id.fill_cavities()
id.find_nasion()
id.check_nasion()
print(id.find_rpa())
print(id.find_lpa())

id.save_pts_to_txt()

id.show_3D_array(id.head, axis=2, pt=(id.registered_lpa[0], id.registered_lpa[2]))
id.show_3D_array(id.head, axis=2, pt=(id.registered_rpa[0], id.registered_rpa[2]))

id.find_registered_lpa_rpa_nasion()

# AJOUTER L'AFFICHAGE DES PTS TROUVÉS





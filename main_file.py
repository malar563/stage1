import dicom2nifti
import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np
import ants
import time
import pandas as pd


class Segmentation:
    """Processing DICOM files, segmenting the head and storing various image masks."""
    
    def __init__(self, dcm_path="DICOM_010/COW_Angio_0.6_Hv36_3", big_output_directory="processed_files", file_number=0, crop="yes"):
        """
        Initializes the Segmentation object by locating or generating a NIfTI file
        from a DICOM folder, and setting up paths and key image attributes.

        This method either loads an existing NIfTI file (cropped or not), or converts
        the provided DICOM folder to a NIfTI image using `dicom2nifti`, optionally cropping it
        based on resolution.

        Parameters
        ----------
        dcm_path : str, optional
            Path to the directory containing DICOM files. Used only if no NIfTI exists yet.
        
        big_output_directory : str, optional
            Root directory where subfolders of NIfTI and mask outputs will be saved.
        
        file_number : int, optional
            Identifier for the processed file. Used to name subfolders and outputs.
        
        crop : any type (default="yes")
            If not None, cropping will be applied after DICOM conversion based on image resolution.

        Sets
        ----
        - self.nii_path : path to the NIfTI image (cropped if found/generated)
        - self.array : numpy array of image data loaded from NIfTI
        - self.resolution : voxel spacing in mm (tuple of 3 floats)
        - self.dimension : image shape (tuple of 3 ints)

        Files Created
        -------------
        - If no NIfTI file is found, runs `self.dcm_to_nii()`:
            - Generates a `.nii.gz` file from the DICOM folder
            - Applies cropping if enabled
        - Automatically calls `self.save_to_csv()` to save image metadata to `points<file_number>.txt`

        Attributes
        ----------
        dcm_path : str
            Input DICOM folder path.
        
        big_output_directory : str
            Root folder for outputs.
        
        file_number : str
            File identifier, formatted as string.
        
        nifti_output_directory : str
            Path to the subfolder for this case’s outputs.
        
        nii_path : str
            Full path to the NIfTI file used as base image.
        
        not_cropped_nii_path : str or None
            Path to the uncropped image, if both exist.
        
        img : nibabel.Nifti1Image
            NIfTI image object corresponding to `nii_path`.
        
        array : np.ndarray
            Image voxel data.
        
        resolution : tuple of float
            Voxel spacing (dy, dx, dz).
        
        dimension : tuple of int
            Image dimensions (Y, X, Z).

        Future attributes
        -----------------
        Masks you can access at one point in the segmentation process :
            -> self.head, self.skull, self.no_arteries_array, self.arteries
            -> self.brain_totalsegmentator and self.skull_totalsegmentator            

        Notes
        -----
        IMPORTANT :
            - In this class, when displaying an array, the index order is (Y, X, Z)
            - According to Nibabel, reorient automatically the image in RAS+ coordinates    
        """

        self.dcm_path = dcm_path # Path to the DICOM directory.
        self.big_output_directory = big_output_directory # Path to the top-level directory for storing outputs.
        self.file_number = str(file_number) # String version of the file number used in folder naming.
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number) # Full path to the directory where NIfTI outputs will be stored.
        

        # Check whether the specified path exists or not
        exist = os.path.exists(self.nifti_output_directory)
        if not exist:
                # If no file is found, generating one from the specified DICOM folder
                print("No NIfTI file found. Processing the specified DICOM folder...")
                self.dcm_to_nii(crop)
        else:
            # Listing NIfTI files in the folder
            nii_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith(".nii") or f.endswith(".nii.gz")]
            # Searching a file starting by "cropped_"
            cropped_files = [f for f in nii_files if f.startswith("cropped_")]
            if cropped_files:
                # If "cropped_" file is found
                self.nii_path = os.path.join(self.nifti_output_directory, cropped_files[0])
                self.not_cropped_nii_path = os.path.join(self.nifti_output_directory, cropped_files[0].removeprefix("cropped_"))
            else:
                # Trying to find a non-cropped file
                excluded_prefixes = ("cropped_", "fwd", "inv", "mask", "mca_territory", "totalsegmentator") 
                all_non_cropped = [f for f in nii_files if not f.startswith(excluded_prefixes)]
                if all_non_cropped:
                    self.nii_path = os.path.join(self.nifti_output_directory, all_non_cropped[0])
                else:
                    # If no file is found, generating one from the specified DICOM folder
                    print("No NIfTI file found. Processing the specified DICOM folder...")
                    self.dcm_to_nii()

        self.img = nib.load(self.nii_path)
        self.array = self.img.get_fdata()
        self.resolution = self.img.header["pixdim"][1:4]
        self.dimension = self.img.shape
        self.save_to_csv()


    def save_to_csv(self):
        """
        Save image metadata to a CSV file. It contains :
        - Voxel dimensions along the x, y, z axes
        - Spatial resolution in millimeters
        - Physical length of the volume in each direction (dimension × resolution)

        Files Created
        -------------
        - points{self.file_number}.txt : CSV file containing metadata.

        Notes
        -----
        - Axes are reordered to (x, y, z) for readability and consistency.
        """
        csv_path = os.path.join(self.nifti_output_directory, f"points{self.file_number}.csv")
        data = [["","x", "y", "z"],
                ["Dimensions", self.dimension[1], self.dimension[0], self.dimension[2]],
                ["Dimensions not cropped", "-", "-", "-"],
                ["Resolution (mm)", self.resolution[1], self.resolution[0], self.resolution[2]],
                ["Length (mm)", self.dimension[1]*self.resolution[1], self.dimension[0]*self.resolution[0], self.dimension[2]*self.resolution[2]]]
        if hasattr(self, 'not_cropped_nii_path'): # check  if the variable exists
            not_cropped_img = nib.load(self.not_cropped_nii_path)
            data[2][1], data[2][2], data[2][3] = not_cropped_img.shape 

        # Ne vas pas marcher si pas de crop (deux lignes ou une à rajouter)
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_existing.values[:5,:4] = np.array(data)
            print(df_existing.values)
            df_existing.to_csv(csv_path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)             


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
        print(nifti_files)
        nifti_path = os.path.join(self.nifti_output_directory, nifti_files[0]) # Use the first .nii.gz file found
        print(f"NIfTI generated : {nifti_path}")

        self.nii_path = nifti_path

        if crop is not None:
            # Load the image with nibabel
            self.not_cropped_nii_path = nifti_path
            nifti_image = nib.load(nifti_path)

            # Crop the image depending on the resolution 
            pix_dim, pix_z = nifti_image.header["pixdim"][1:4], nifti_image.header["pixdim"][3]
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

            self.nii_path = nifti_path

   
    def show_3D_array(self, arr, axis=0, pts=None):
        """
        Display a 3D array slice-by-slice using a matplotlib slider.

        Arguments
        ----------
        arr : numpy.ndarray
            3D array to visualize.

        axis : int
            Axis along which to slice.

        pts : list of ((x, y), slice_idx, color) tuples
            Points to highlight on specific slices, each with a custom color.
        """
        from matplotlib.widgets import Slider
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        index = arr.shape[axis] // 2

        if axis == 0:
            img = ax.imshow(arr[index, :, :], cmap="gray", origin="lower")
        elif axis == 1:
            img = ax.imshow(arr[:, index, :], cmap="gray", origin="lower")
        else:
            img = ax.imshow(arr[:, :, index], cmap="gray", origin="lower")

        # Initialize scatter plot (empty)
        point_plot = ax.scatter([], [], c=[], marker='o', s=80, edgecolors='none')

        def update(val):
            slice_idx = int(slice_slider.val)

            # Update image
            if axis == 0:
                img.set_data(arr[slice_idx, :, :])
            elif axis == 1:
                img.set_data(arr[:, slice_idx, :])
            else:
                img.set_data(arr[:, :, slice_idx])

            # Filter points on current slice
            if pts is not None:
                current_pts = [(xy, color) for (xy, sl, color) in pts if sl == slice_idx]
                if current_pts:
                    coords = [xy for xy, _ in current_pts]
                    colors = [c for _, c in current_pts]
                    point_plot.set_offsets(coords)
                    point_plot.set_color(colors)
                else:
                    point_plot.set_offsets(np.empty((0, 2)))
                    point_plot.set_color([])
            else:
                point_plot.set_offsets(np.empty((0, 2)))
                point_plot.set_color([])

            fig.canvas.draw_idle()

        # Slider
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slice_slider = Slider(ax_slider, 'Slice', 0, arr.shape[axis] - 1, valinit=index, valstep=1)
        slice_slider.on_changed(update)

        update(index)
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
        - totalsegmentator{file_number}.nii.gz : Segmented head     

        Notes
        -----
        ONLY RUNS WHEN THE SCRIPT IS EXECUTED DIRECTLY.
        Important : 
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
            output_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii.gz")
            nib.save(output_img, output_path)
            print(f"NIfTI generated : {output_path}.nii.gz")


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
        totalsegmentator_mask = nib.load(os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii.gz")).get_fdata()
        self.brain_totalsegmentator = np.where(totalsegmentator_mask == 90, 1, 0)
        self.skull_totalsegmentator = np.where(totalsegmentator_mask == 91, 1, 0) 
        self.arteries = self.brain_totalsegmentator * self.arteries

    
    def mask_to_nii(self, iter_erosion=3):
        """
        Save head and masks as NIfTI files.
        Perform the final refinement of the skull mask using the output generated by TotalSegmentator.

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
        - head{file_number}.nii.gz : Non-masked head image (used for registration). 
        - mask{file_number}.nii.gz : Labeled mask
        """
        # Nifti file with the HU units of the whole head for the registration
        head_array = np.where(self.head == 1, self.array, -1000) # HU units where the mask is 1, -1000 where the mask is 0
        head_image = nib.Nifti1Image(head_array, self.img.affine, self.img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz")
        nib.save(head_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}.nii.gz")

        total_mask = 1*self.head + 2*self.skull + 1*self.arteries

        # Final improvement of the mask of the skull
        from scipy.ndimage import binary_erosion, generate_binary_structure
        mask_soft_tissues = np.where(total_mask == 1, 1, 0)
        mask_eroded_skull = binary_erosion(self.skull_totalsegmentator, structure=generate_binary_structure(3,1), iterations=iter_erosion)
        not_included_skull = 3*mask_soft_tissues*mask_eroded_skull
        self.skull = self.skull + (mask_soft_tissues*mask_eroded_skull)
        total_mask = total_mask + not_included_skull

        # NIfTI file of the mask
        masked_image = nib.Nifti1Image(total_mask, self.img.affine, self.img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii.gz")
        nib.save(masked_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}.nii.gz")


# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
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
            - In this class, when displaying an array, note that the index order is 
              (Z, X, Y) and NOT (Y, X, Z) as it was in the Segmentation class.
              It is to ensure consistency of the orientation (for both fixed and moving image)
              and a better display.
            - To get the points in the original coordinate system (mask) :
                -> For a mask, use ants.reorient_image2(image, orientation)
                -> For a point, use a affine matrix as found in the 
        """
        
        self.big_output_directory = big_output_directory
        self.file_number = str(file_number)
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number)
        self.moving_img_path = os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz")
        self.fixed_img_path = fixed_img_path

        self.initial_moving_img = ants.image_read(self.moving_img_path) # Image provided by the mask
        self.initial_moving_img_orientation = ants.get_orientation(self.initial_moving_img) # Orientation of the mask
        self.orient_for_registration = "IAL" # Chosen orientation (easier to visualize). Do not change it : all the following code rely on this
        self.moving_img = ants.image_read(self.moving_img_path, reorient=self.orient_for_registration) # Image to be registered
        self.fixed_img = ants.image_read(fixed_img_path, reorient=self.orient_for_registration) # Reference image for registration

        # Landmark coordinates determined visually in the normalized space
        self.lpa_vox_normal_space = np.array([25, 107, 6])
        self.rpa_vox_normal_space = np.array([25, 107, 173])
        self.nas_vox_normal_space = np.array([28, 4, 90])

        self.filled_y_slices = [] # Used to determine positions (lpa and rpa)
        mask_img = nib.load(os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii.gz"))
        self.mask_dimension = mask_img.shape
        self.head = mask_img.get_fdata() 
        # IMPORTANT : from now on, it is [z,x,y] and not [y,x,z] as it was in the Segmentation class
        self.filled_head = self.reorient_point_to_original_mask(object_to_reorient=np.where(self.head>=1, 1, 0), 
                                                                initial_orient=self.initial_moving_img_orientation, 
                                                                final_orient=self.orient_for_registration)
        self.head = np.copy(self.filled_head)

    
    def reorient_point_to_original_mask(self, object_to_reorient=None, initial_orient="IAL", final_orient="RPI"):
        """
        Reorients a 3D array or a 3D point from one coordinate system to another 
        by applying axis transpositions and flips based on orientation codes.

        Parameters
        ----------
        object_to_reorient : np.ndarray or tuple
            The object to reorient. Can be either:
            - A 3D NumPy array (volume) with shape,
            - A 3-element tuple/list or dictionary representing a 3D point.

        initial_orient : str
            A 3-letter orientation code representing the current axis directions 
            (e.g., "IAL" = Inferior, Anterior, Left).

        final_orient : str
            A 3-letter orientation code representing the desired axis directions 
            (e.g., "RPI" = Right, Posterior, Inferior).

        Returns
        -------
        object_reoriented : same type as input
            - If a volume is passed, returns the array reoriented (flipped and/or transposed).
            - If a point is passed, returns the coordinates mapped to the new orientation system.

        Notes
        -----
        - Orientation codes use the anatomical directions: 
          'R'=Right, 'L'=Left, 'A'=Anterior, 'P'=Posterior, 'I'=Inferior, 'S'=Superior.
        - Axis flips are performed if the direction in the initial orientation is opposite 
          to the one in the target orientation.
        - Axis reordering is performed if axes are permuted between orientations.
        - This method assumes the internal attribute `self.mask_dimension` is defined 
          when flipping a point.
        """
        # Initialization
        initial_orient = list(initial_orient)
        final_orient = list(final_orient)
        dict_orientation = {"R":"L", "L":"R", "A":"P", "P":"A", "I":"S", "S":"I"}
        transpose_needed = []
        already_in = ["N", "N", "N"]

        # Determining transforms to apply
        for initial_axis, letter in enumerate(initial_orient):
            try:
                already_in[initial_axis]= (initial_axis, final_orient.index(letter)) # If already in, (initial axis number, final axis number)
            except:
                transpose_needed.append((letter, initial_axis)) # Initial axis where the letter needs to be changed
                new_letter = dict_orientation[letter]
                initial_orient[initial_axis] = new_letter
        for initial_axis, element in enumerate(initial_orient):
            already_in[initial_axis] = (initial_axis, final_orient.index(element))

        # Reorienting a volume
        if isinstance(object_to_reorient, np.ndarray) and len(object_to_reorient.shape)==3:
            for element, axis_to_flip in transpose_needed:
                object_to_reorient = np.flip(object_to_reorient, axis=axis_to_flip)
            object_to_reorient = np.transpose(object_to_reorient, (already_in[0][1], already_in[1][1], already_in[2][1]))
            return object_to_reorient

        # Reorienting a point
        elif isinstance(object_to_reorient, (np.ndarray, dict, tuple)):
            initial_point = object_to_reorient[0], object_to_reorient[1], object_to_reorient[2], 1
            affine = np.zeros((4,4))
            affine[0,already_in[0][1]] = 1
            affine[1,already_in[1][1]] = 1
            affine[2,already_in[2][1]] = 1
            affine[3,3] = 1
            for letter, initial_index in transpose_needed:
                new_index = already_in[initial_index][1]
                affine[new_index] = -1*affine[new_index]
                affine[new_index,-1] = self.mask_dimension[initial_index]-1
            # print(affine)
            new_point = (affine @ initial_point)[:-1]
            return new_point
        
        # Dealing with exeptions
        else:
            print("Invalid object")    


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
        print('registered LPA', self.registered_lpa) # (z,x,y)

        # RPA : automatically identify on patient
        rpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.rpa_vox_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, rpa_pt_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, rpa_pt_patient_space)
        rpa_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, rpa_pt_patient_space)
        self.registered_rpa = round(rpa_vox_patient_space[0]), round(rpa_vox_patient_space[1]), round(rpa_vox_patient_space[2])
        print('registered RPA', self.registered_rpa) # (z,x,y)

        # nasion : automatically identify on patient
        nas_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.nas_vox_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, nas_pt_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, nas_pt_patient_space)
        nas_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, nas_pt_patient_space)
        self.registered_nasion = round(nas_vox_patient_space[0]), round(nas_vox_patient_space[1]), round(nas_vox_patient_space[2])
        print('registered nasion', self.registered_nasion) # (z,x,y)


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
        """        
        from scipy.ndimage import binary_fill_holes

        for i in range(0, len(self.filled_head[1,1,:])):
            # Fill holes in the x-z and y-z plane 
            original = self.filled_head[:,:,i]
            filled = binary_fill_holes(self.filled_head[:,:,i])
            if not np.array_equal(original,filled): # Check if the slice has been filled
                self.filled_y_slices.append(i)
            self.filled_head[:,:,i] = filled
        for i in range(0, len(self.filled_head[1,:,1])):
            self.filled_head[:,i,:] = binary_fill_holes(self.filled_head[:,i,:]) 
        # print(self.filled_y_slices)
        for i in range(0, len(self.filled_head[:,1,1])):
            # Fill holes in the x-y plane
            self.filled_head[i,:,:] = binary_fill_holes(self.filled_head[i,:,:])
    

    def find_nasion(self, window=5):
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
        self.show_3D_array(ROI_nas)
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(ROI_nas, axis = 1) # axis=2 sums in y, axis=0 sums in z

        # Index of the maximal values for rows and minimal values for columns
        max_index_row = (np.argmin(counts_x, axis=0))
        min_index_column = (np.argmax(counts_x, axis=1))
        print(max_index_row, min_index_column)

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
        # nasion_x_final = ((2*window) - counts_x[int(np.mean(nasion_row)),int(np.mean(nasion_column))]) + (reg_nas_x - window)

        if np.any(nasion_row == 0) or np.any(nasion_row == (window*2) - 1):
            nasion_y_final = reg_nas_y
        if np.any(nasion_column == 0) or np.any(nasion_column == (window*2) - 1):
            nasion_z_final = reg_nas_z

        nasion_x_final = np.nonzero(self.filled_head[nasion_z_final,:,nasion_y_final])[0][0]
        print(nasion_z_final, nasion_x_final, nasion_y_final)
        self.nasion = nasion_z_final, nasion_x_final, nasion_y_final # (z, x, y)


    def check_nasion(self):
        """
        Visually validate the refined nasion position against the initially registered one.

        This function plots axial, coronal, and sagittal views of the head mask, overlaying
        both the refined nasion (in red) and the registered nasion (in blue) to assess the
        accuracy of the correction.
        """
        # # x-axis goes through the nasion
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


        

    def find_depth_rpa(self):
        nonzero = np.nonzero(self.filled_head[self.registered_rpa[0],self.registered_rpa[1],:])[0]
        if self.registered_rpa[2] in nonzero:
            print(nonzero)   
            index_rpa_y = np.where(nonzero == self.registered_rpa[2])[0][0]
            print(index_rpa_y)
            for i in range(1, len(nonzero)-index_rpa_y):
                if nonzero[index_rpa_y+i] != self.registered_rpa[2]+i:
                    return nonzero[index_rpa_y+i-1] # gets the surface of the head
            return nonzero[-1] # gets the surface of the head
        else:
            index_lpa = np.argmin(np.abs(nonzero-self.registered_rpa[2]))
            return nonzero[index_lpa] # gets the surface of the head
        


    def find_depth_lpa(self):
        nonzero = np.nonzero(self.filled_head[self.registered_lpa[0],self.registered_lpa[1],:])[0]
        if self.registered_lpa[2] in nonzero:
            print(nonzero)   
            index_lpa_y = np.where(nonzero == self.registered_lpa[2])[0][0]
            print(index_lpa_y)
            for i in range(1, index_lpa_y):
                if nonzero[index_lpa_y-i] != self.registered_lpa[2]-i:
                    return nonzero[index_lpa_y-i+1] # gets the surface of the head
            return nonzero[0] # gets the surface of the head
        else:
            index_lpa = np.argmin(np.abs(nonzero-self.registered_lpa[2]))
            return nonzero[index_lpa] # gets the surface of the head
        
    
    def improve_lpa_rpa(self):
        self.lpa = self.registered_lpa[0], self.registered_lpa[1], self.find_depth_lpa()
        self.rpa = self.registered_rpa[0], self.registered_rpa[1], self.find_depth_rpa()





    def save_pts_to_csv(self):
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
        csv_path = os.path.join(self.nifti_output_directory, f"points{self.file_number}.csv")

        df_existing = pd.read_csv(csv_path)
        print(df_existing)
        df_to_keep = df_existing.values[:5,:4].tolist()
        print(df_to_keep)
        improved_landmarks = [self.nasion, self.lpa, self.rpa]
        registered_landmarks = [self.registered_nasion, self.registered_lpa, self.registered_rpa]
        for i, point in enumerate(improved_landmarks):
            improved_landmarks[i] = self.reorient_point_to_original_mask(point,
                                                              initial_orient=self.orient_for_registration,
                                                              final_orient=self.initial_moving_img_orientation)
        for i, point in enumerate(registered_landmarks):
            registered_landmarks[i] = self.reorient_point_to_original_mask(point,
                                                              initial_orient=self.orient_for_registration,
                                                              final_orient=self.initial_moving_img_orientation)
        print(improved_landmarks)
        data = [["Nasion improved (voxel)", improved_landmarks[0][1], improved_landmarks[0][2], improved_landmarks[0][0]],
                ["Nasion registered (voxel)", registered_landmarks[0][1], registered_landmarks[0][2], registered_landmarks[0][0]],
                ["LPA improved (voxel)", improved_landmarks[1][1], improved_landmarks[1][2], improved_landmarks[1][0]],
                ["LPA registered (voxel)", registered_landmarks[1][1], registered_landmarks[1][2], registered_landmarks[1][0]],
                ["RPA improved (voxel)", improved_landmarks[2][1], improved_landmarks[2][2], improved_landmarks[2][0]],
                ["RPA registered (voxel)", registered_landmarks[2][1], registered_landmarks[2][2], registered_landmarks[2][0]],
                ["If '-' appears in the line 'Dimensions not cropped', it means the original (uncropped) NIfTI file was used for the entire process."],
                ["If the cropped NIfTI file was used to retrieve the original coordinates of the LPA RPA and Nasion : subtract the z-dimension of the cropped file from that of the original file, and add the difference to the z-index. The x and y indices remain unchanged."]]
        df_to_keep += data
        print(df_to_keep)
        df = pd.DataFrame(df_to_keep)
        df.to_csv(csv_path, index=False)
        



    def delete_useless_files(self):
        """
        Deletes intermediate NIfTI files from the output directory.

        Specifically, removes:
        - The original uncropped NIfTI file (if a cropped version is used)
        - The `totalsegmentator` output file
        - The full-head image used for registration
        - The forward and inverse transform files

        This helps reduce disk usage and declutter the output folder after segmentation
        and registration steps are completed.

        Notes
        -----
        - If a file does not exist, a message is printed instead of raising an error.
        - Safe to call even if some files are missing.
        IMPORTANT :
            Must be used AFTER the registration is done
        """
        # Listing NIfTI files in the folder
        useless_files = [os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat"),
                         os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz"),
                         os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat"),
                         os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz"),
                         os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii.gz"),
                         os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz")]
        
        # Finds if a cropped file exists, and delete the original one in that case
        nii_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith(".nii") or f.endswith(".nii.gz")]
        cropped_files = [f for f in nii_files if f.startswith("cropped_")]
        if len(cropped_files)>=1:
            excluded_prefixes = ("cropped_", "fwd", "inv", "mask", "mca_territory", "totalsegmentator") 
            non_cropped = [f for f in nii_files if not f.startswith(excluded_prefixes)][0]
            useless_files.append(os.path.join(self.nifti_output_directory,non_cropped))
        
        for file_path in useless_files:
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"The file {file_path} does not exist")
    

    def mca_territory_mask(self, arterial_territories_path="mni_vascular_territories.nii.gz"):
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

        # Getting the arteries mask
        self.arteries = ants.image_read(os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii.gz"), reorient="IAL")
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
        counts_z = np.sum(patient_mca_arteries.numpy(), axis = 0) # axis=2 sums in y, axis=0 sums in z
        plt.imshow(counts_z, origin="lower")
        plt.show()

        # Nifti file with the MCA territories
        head_img =  nib.load(os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz"))
        patient_mca_arteries = ants.reorient_image2(patient_mca_arteries, orientation=self.initial_moving_img_orientation) # Reorient accordinf to the initial mask
        mca_image = nib.Nifti1Image(patient_mca_arteries.numpy(), head_img.affine, head_img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "mca_territory"+self.file_number+".nii.gz")
        nib.save(mca_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}.nii.gz")


                



# dicoms_list = ["DICOM_003/Carotid_Angio_0.625mm", "DICOM_010/COW_Angio_0.6_Hv36_3"]
dicoms_list = ["online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463", "online_patient/2.16.840.1.114274.1818.528945204283203896414435929150802789774", "online_patient/2.16.840.1.114274.1818.56920369040074765021783555636978216368"]
# dicoms_list = ["online_patient/test", "2.16.840.1.114274.1818.528945204283203896414435929150802789774", "2.16.840.1.114274.1818.56920369040074765021783555636978216368"]
dicoms_list = ["ct_enligne/1", "ct_enligne/2","ct_enligne/4","ct_enligne/5", "ct_enligne/6", "ct_enligne/7"]
dicoms_list = ["ct_enligne/6", "ct_enligne/7"]

# CHECKER LES AXES PARTOUT POUR ÊTRE SÛR QUE C'EST CHILL
def main(dicoms_list = dicoms_list):
    for i, dicom in enumerate(dicoms_list):
        start = time.time()
        ct = Segmentation(dcm_path=dicom, big_output_directory="ct_enligne_nifti", file_number=i+4)
        ct.apply_threshold()       
        ct.keep_largest_island()
        ct.fill_holes()
        ct.remove_arteries()

        # Totalsegmentator
        ct.segment_brain()
        ct.arteries_and_totalsegmentator_mask()
        ct.mask_to_nii()

        # ct.show_3D_array(ct.skull, axis=2) # En z
        # ct.show_3D_array(ct.head, axis=0, pt=(50,42), pt_slice = 100)
        id = Registration(big_output_directory="ct_enligne_nifti", file_number=i+4, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

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


# if __name__ == "__main__":
#     main()








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



# 6_cta_thins
id = Registration(big_output_directory="online_angio", file_number=2, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

# id.register(show=True)
id.read_transforms()

id.find_registered_lpa_rpa_nasion()
id.fill_cavities()
id.find_nasion(window=5)
id.check_nasion()
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





    # id.show_3D_array(id.moving_img.numpy())
    # id.show_3D_array(id.fixed_img.numpy())

    # id.show_3D_array(nib.load('icbm_avg_152_t1_tal_lin.nii').get_fdata(), axis=0)
    # id.show_3D_array(nib.load('jspakoi/0/head0.nii').get_fdata(), axis=0)
    # id.show_3D_array(id.head, axis=0)
    # id.show_3D_array(id.moving_img.numpy(), axis=0)
    # id.show_3D_array(id.fixed_img.numpy(), axis=0)

    # id.show_3D_array(id.head, axis=2, pt=(id.registered_lpa[0], id.registered_lpa[2]))
    # id.show_3D_array(id.head, axis=2, pt=(id.registered_rpa[0], id.registered_rpa[2]))



    # AJOUTER L'AFFICHAGE DES PTS TROUVÉS





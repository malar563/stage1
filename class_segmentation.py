import dicom2nifti
import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess


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
            self.dcm_to_nii(crop=True)
        else:
            # Listing NIfTI files in the folder
            nii_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith(".nii") or f.endswith(".nii.gz")]
            # Searching a file starting by "cropped_"
            cropped_files = [f for f in nii_files if f.startswith("cropped_")]
            if cropped_files:
                # If "cropped_" file is found
                self.nii_path = os.path.join(self.nifti_output_directory, cropped_files[0])
                self.not_cropped_nii_path = os.path.join(self.nifti_output_directory, cropped_files[0].removeprefix("cropped_"))
                print("NIfTI file found. No DICOM processing will be done.")
            else:
                # Trying to find a non-cropped file
                excluded_prefixes = ("cropped_", "fwd", "inv", "mask", "mca_territory", "totalsegmentator") 
                all_non_cropped = [f for f in nii_files if not f.startswith(excluded_prefixes)]
                if all_non_cropped:
                    self.nii_path = os.path.join(self.nifti_output_directory, all_non_cropped[0])
                    print("NIfTI file found. No DICOM processing will be done.")
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
        - points{self.file_number}.csv : CSV file containing metadata.

        Notes
        -----
        - Axes are reordered to (x, y, z) for readability and consistency.
        """
        csv_path = os.path.join(self.nifti_output_directory, f"points{self.file_number}.csv")

        points_csv_file = [f for f in os.listdir(self.nifti_output_directory) if f.endswith(f"points{self.file_number}.csv")]
        if points_csv_file:
            os.remove(csv_path)

        data = [[self.dcm_path,"x", "y", "z"],
                ["Dimensions", self.dimension[1], self.dimension[0], self.dimension[2]],
                ["Dimensions not cropped", "-", "-", "-"],
                ["Resolution (mm)", self.resolution[1], self.resolution[0], self.resolution[2]],
                ["Length (mm)", self.dimension[1]*self.resolution[1], self.dimension[0]*self.resolution[0], self.dimension[2]*self.resolution[2]]]
        if hasattr(self, 'not_cropped_nii_path'): # check  if the variable exists
            not_cropped_img = nib.load(self.not_cropped_nii_path)
            data[2][1], data[2][2], data[2][3] = not_cropped_img.shape 

        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_existing.values[:5,:4] = np.array(data)
            df_existing.to_csv(csv_path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)             


    def dcm_to_nii(self, crop=True, size_head=250):
        """
        Convert a DICOM series to a NIfTI file and optionally crop it along the z-axis.

        Uses `dicom2nifti` to convert DICOM files in `self.dcm_path` to a `.nii.gz` file 
        saved in `self.nifti_output_directory`. If `crop` is enabled, the image is cropped 
        based on voxel spacing along the z-axis: the last 256 or 512 slices are kept 
        depending on slice thickness.

        Parameters
        ----------
        crop : str or None, optional
            If not None, crop the volume depending on slice thickness (z-resolution).

        Sets
        ----    
        self.nii_path : str
            Full path to the primary (possibly cropped) NIfTI file after conversion.  

        self.not_cropped_nii_path : str
            Path to the original, full NIfTI file before cropping. Won't exist if no cropped file was produced

        Files Created
        -------------
        - <generated_name>.nii.gz : Converted DICOM to NIfTI
        - cropped_<generated_name>.nii.gz : Cropped version saved if cropping is enabled
        """

        # Create the output_directory file
        os.makedirs(self.nifti_output_directory, exist_ok=True)

        # # Convert DICOM to NIfTI (compression=False -> .nii instead of .nii.gz)
        command = [
            "python",
            "-m", "dcm2niix",
            "-z", "y",
            "-f", "%p_%s",
            "-o", self.nifti_output_directory,
            self.dcm_path]

        convert = subprocess.run(command) # , capture_output=True, text=True run avec : pas d'écritures

        # Find the generated file in the output folder
        nifti_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith('.nii.gz')]
        print(nifti_files)
        nifti_path = os.path.join(self.nifti_output_directory, nifti_files[0]) # Use the first .nii.gz file found
        print(f"NIfTI generated : {nifti_path}")

        self.nii_path = nifti_path

        if crop:
            # Load the image with nibabel
            self.not_cropped_nii_path = nifti_path
            nifti_image = nib.load(nifti_path)
            top_head_bottom_node_distance = 0.75*size_head

            # Crop the image depending on the resolution 
            pix_dim, pix_z = nifti_image.header["pixdim"][1:4], nifti_image.header["pixdim"][3]
            # cropped_data = nifti_image.get_fdata()[:,:,-1*int(top_head_bottom_node_distance/pix_z):]
            n_slices = nifti_image.shape[2]
            crop_start = max(0, n_slices - int(top_head_bottom_node_distance / pix_z))
            cropped_data = nifti_image.get_fdata()[:, :, crop_start:]


            # if pix_z >= 0.6:
            #     cropped_data = nifti_image.get_fdata()[:,:,-256:]
            # else:
            #     cropped_data = nifti_image.get_fdata()[:,:,-512:]

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

        Parameters
        ----------
        arr : numpy.ndarray
            3D image or mask volume to visualize (shape: [D1, D2, D3]).

        axis : int, default=0
            Axis along which to slice (0 = first axis, 1 = second, 2 = third).

        pts : list of tuples ((x, y), slice_idx, color), optional
            Points to highlight on specific slices. Each point is defined by:
                - (x, y) coordinates in the 2D slice
                - slice index (along selected axis)
                - color (any matplotlib-compatible color string)

        Notes
        -----
        - The array is visualized using `imshow`, with slices taken along the specified axis.
        - Points are only shown on the slice corresponding to their `slice_idx`.
        - Points are displayed as circles with customizable color.
        - Axes are displayed in (row, column) order, with `origin="lower"` for intuitive orientation.
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
        import torch

        device = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {device}")

        input_img = nib.load(self.nii_path)
        if only_brain:
            output_img = totalsegmentator(input_img, fast=fast, roi_subset=["brain"], device=device)
        else:
            output_img = totalsegmentator(input_img, fast=fast, device=device)
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
        Save the segmented head and associated masks as NIfTI files.

        This function generates:
        1. A NIfTI image of the original scan masked by the head (used for registration).
        2. A labeled mask volume combining:
            - Air (value = 0)
            - Head (value = 1)
            - Arteries (value = 2)
            - Skull (value = 3), refined using TotalSegmentator output and soft tissue overlap.

        The skull mask is improved by overlapping the actual skull mask with the skull mask
        from TotalSegmentator.

        Parameters
        ----------
        iter_erosion : int
            Number of binary erosion iterations to apply on the TotalSegmentator skull mask
            before adding it to the main mask as refined skull.

        Files Created
        -------------
        - head{file_number}.nii.gz : Original scan with non-head voxels set to -1000 HU (for registration).
        - mask{file_number}.nii.gz : Integer-labeled volume with values 0–3 indicating tissue types.

        Notes
        -----
        - The skull refinement uses logical operations: refined regions are added only where 
          soft tissue overlaps the eroded TotalSegmentator skull.
        - The resulting mask is used in further processing such as artery registration.
        """
        # Nifti file with the HU units of the whole head for the registration
        head_array = np.where(self.head == 1, self.array, -1000) # HU units where the mask is 1, -1000 where the mask is 0
        head_image = nib.Nifti1Image(head_array, self.img.affine, self.img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz")
        nib.save(head_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}")

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
        print(f"NIfTI generated : {nifti_path}")
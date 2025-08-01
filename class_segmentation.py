import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess


class Segmentation:
    """Processing DICOM files, segmenting the head and storing various image masks."""
    
    def __init__(self, dcm_path="DICOM_010/COW_Angio_0.6_Hv36_3", big_output_directory="processed_files", file_number=0, crop=True):
        """
        This method either loads an existing NIfTI file (cropped or not), or converts
        the provided DICOM folder to a NIfTI image using `dcm2niix`, optionally cropping it
        based on resolution.

        It also sets up paths and image attributes.

        Parameters
        ----------
        dcm_path : str, optional
            Path to the directory containing DICOM files. Used only if no NIfTI exists yet.
        
        big_output_directory : str, optional
            Root directory where subfolders of NIfTI and mask outputs will be saved.
        
        file_number : int, optional
            Identifier (number) for the processed file. Used to name subfolders and outputs.
        
        crop : bool (default=True)
            If True, cropping will be applied after DICOM conversion based on image resolution.

        Sets
        ----
        - self.nii_path : path to the NIfTI image (cropped if found/generated)
        - self.array : numpy array of image data loaded from NIfTI
        - self.resolution : voxel spacing in mm (tuple of 3 floats)
        - self.dimension : image shape (tuple of 3 ints)

        Files Created
        -------------
        - If no NIfTI file is found in the folder, runs `self.dcm_to_nii()`:
            - Generates a `.nii.gz` file from the DICOM folder in nifti_output_directory
            - Applies cropping if enabled
        - Calls `self.save_to_csv()` in all cases to save image metadata to `points<file_number>.csv`

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
            Path to the uncropped image, if both cropped and uncropped files exist.
        
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
            self.dcm_to_nii(crop=crop)
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
        Save image metadata to a CSV file named `points{file_number}.csv`.

        The CSV contains:
        - Input DICOM path
        - Voxel dimensions along the x, y, z axes (image shape)
        - Spatial resolution in millimeters (voxel spacing)
        - Physical length of the cropped volume in each direction (dimension × resolution)
        - Uncropped voxel dimensions, if available

        Files Created
        -------------
        - points{self.file_number}.csv : CSV file containing image dimensions and resolutions.
          If the file already exists, it is deleted before creating a new one.

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
        Convert a DICOM series to a NIfTI file using `dcm2niix`, and optionally crop it along the z-axis.

        This function calls `dcm2niix` to convert the contents of `self.dcm_path` into a `.nii.gz` 
        file stored in `self.nifti_output_directory`. If `crop=True`, it keeps approximately 
        the last 75% of a head volume (based on a default physical size of 250 mm) in the z-direction, 
        using the image's voxel size to calculate how many slices to retain.

        Parameters
        ----------
        crop : bool, optional
            Whether to crop the image along the z-axis after conversion. Defaults to True.

        size_head : float, optional
            Estimated physical height of the head in millimeters, used to determine how many 
            slices to keep during cropping. Defaults to 250 mm.

        Sets
        ----
        self.nii_path : str
            Full path to the final NIfTI file used in the pipeline (cropped if applicable).

        self.not_cropped_nii_path : str
            Path to the original full-volume NIfTI file before cropping, only set if cropping is performed.

        Files Created
        -------------
        - <generated_name>.nii.gz :
            NIfTI file converted from the DICOM series.

        - cropped_<generated_name>.nii.gz :
            Cropped NIfTI file saved if cropping is enabled.

        Notes
        -----
        - Cropping is performed by retaining the last N slices where N ≈ 0.75 × size_head ÷ slice_thickness.
        - The header and affine from the original NIfTI are preserved in the cropped version.
        """

        # Create the output_directory file
        os.makedirs(self.nifti_output_directory, exist_ok=True)

        # # Convert DICOM to NIfTI
        command = [
            "python",
            "-m", "dcm2niix",
            "-z", "y",
            "-f", "%p_%s",
            "-o", self.nifti_output_directory,
            self.dcm_path]

        convert = subprocess.run(command) 

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
        Display a 3D volume slice-by-slice using a matplotlib slider.

        A specific axis is selected for 2D slicing through the volume`. 
        Optional points can be overlaid as colored circles on specific slices.

        Parameters
        ----------
        arr : numpy.ndarray
            3D array to visualize (typically an image or mask).
            Shape must be (D1, D2, D3), where slicing is done along one of the axes.

        axis : int, default=0
            Axis along which to slice the volume:
            - 0 → sagittal-like slicing (ZX)
            - 1 → coronal-like slicing (YZ)
            - 2 → axial-like slicing (XY)
            Note : These axis number are only valid for the class Segmentation.

        pts : list of tuples ((x, y), slice_idx, color), optional
            List of points to highlight on individual slices.
            Each entry is a tuple of:
                - (x, y): coordinates in the 2D slice (row, col)
                - slice_idx: index of the slice along the selected axis
                - color: a matplotlib-compatible color string (e.g., "red", "#00ff00")

        Notes
        -----
        - Uses matplotlib's `Slider` widget for interactive navigation through slices.
        - Slices are shown using `imshow` with `origin='lower'` for intuitive display.
        - Highlighted points are drawn only on their corresponding `slice_idx`.
        - Axes are ordered as (row, column) for `imshow`, which may differ from the actual 
          axis order in the array depending on the selected slicing direction.

        Example
        -------
        self.show_3D_array(self.head, axis=2, pts=[((100, 100),100, "red"), ((50, 50),50, "green")])
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

        Generates 3D binary masks (0 or 1) for head, skull, arteries, and a no-arteries region 
        based on fixed intensity thresholds applied to `self.array`.

        Arguments
        ----------
        threshold_head : int
            Lower bound for head tissue detection.
        
        threshold_skull : int
            Lower bound for skull segmentation.
        
        threshold_no_arteries : int
            Threshold above which regions are considered free of arteries.
            Only high HU value bones should be left.
        
        threshold_arteries : int
            Lower bound for artery detection.
            As much artery regions as possible must be taken.

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
        (e.g., remove metal frame artifacts or isolated noise).

        Applies 3D connected component labeling to `head`, `skull` and `no_arteries_array`,
        and retains only the largest region in each.
        Applies 3D connected-component labeling to the existing masks (`self.head`, `self.skull`, and
        `self.no_arteries_array`) and retains only the largest region in each while ignoring the background (0).

        Updates
        -------
        self.head : numpy.ndarray
            Binary mask of the where only the biggest component (head) is left.
        self.skull : numpy.ndarray
        self.no_arteries_array : numpy.ndarray

        Notes
        -----
        - Connectivity uses a full 3×3×3 structuring element (maximum adjacency).
        - Will only work to remove a metal frame if it is not connected with the head.
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
        Remove arteries from the skull mask (containing arteries) 
        based on proximity to artery-free regions (skull with high HU value).

        Uses a distance transform on `no_arteries_array` to exclude skull voxels 
        too far from artery-free regions, then slightly dilates the result.

        Arguments
        ----------
        max_distance : int
            Maximum distance (in voxels) to keep in the skull mask 
            around the mask of the skull with high HU value region.

        Updates
        -------
        self.skull : numpy.ndarray
            Skull mask with distant (likely artery) parts removed, then slightly dilated.
            It also remove bones with low HU value that are too far from the high HU value region,
            but a correction will be made with the TotalSegmentator mask later.
        """
        from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure

        self.no_arteries_array = self.no_arteries_array != 1
        distance = distance_transform_edt(self.no_arteries_array) # Compute distance of the voxel from closet 0 value
        close_to_bone = distance < max_distance 
        self.skull = self.skull & close_to_bone # Voxels in the skull mask must be in the original mask AND close to higher HU value bones
        self.skull = binary_dilation(self.skull, generate_binary_structure(3, 1))


    def segment_brain(self, fast=False):
        """
        Run TotalSegmentator to segment the head (brain and skull mainly).

        Loads the NIfTI image from `self.nii_path` and performs segmentation using 
        TotalSegmentator.

        Argument
        --------
        fast : bool
            If True, uses lower resolution (3mm instead of 1.5mm) for faster segmentation. 
            (not recommanded for our purposes)

        File Created
        -------------
        - totalsegmentator{file_number}.nii.gz : Segmented head     

        Notes
        -----
        ONLY RUN THIS SCRIPT IN A <<if __name__ == "__main__">> environment.
        - The code chooses GPU if available, otherwise CPU.
        - If the output file already exists, segmentation is skipped.
        Important : 
            Brain is labeled with the number 90
            Skull is labeled with the number 91
        """

        segmentator_nii_file = [f for f in os.listdir(self.nifti_output_directory) if f.endswith(f"totalsegmentator{self.file_number}.nii.gz")]
        if not segmentator_nii_file:
            import torch

            device = "gpu" if torch.cuda.is_available() else "cpu"
            print(f"Running on device: {device}")

            input_img = nib.load(self.nii_path)
            output_img = totalsegmentator(input_img, fast=fast, device=device)
            print("Segmentation with TotalSegmentator has been completed")
            output_path = os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii.gz")
            nib.save(output_img, output_path)
            print(f"NIfTI generated : {output_path}")
        else:
            print(f"totalsegmentator{self.file_number}.nii.gz already exists. The existing one will be taken.")


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
            The artery mask doesn't include soft tissues outside of the brain region anymore.
        """
        totalsegmentator_mask = nib.load(os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii.gz")).get_fdata()
        self.brain_totalsegmentator = np.where(totalsegmentator_mask == 90, 1, 0)
        self.skull_totalsegmentator = np.where(totalsegmentator_mask == 91, 1, 0) 
        self.arteries = self.brain_totalsegmentator * self.arteries

    
    def mask_to_nii(self, iter_erosion=3):
        """
        Save the segmented head and associated masks as NIfTI files.

        This function generates:
        1. head{file_number}.nii.gz : A NIfTI image of the original scan masked by the head 
                                      (used for registration).
        2. mask{file_number}.nii.gz: integer-labeled volume with the following labels:
            0 = background / air
            1 = head (soft tissue)
            2 = arteries
            3 = skull (original)
            4 = refined skull (where eroded TotalSegmentator skull overlaps soft tissue)
            Note : For the skull mask, keep values >= 3

        Parameters
        ----------
        iter_erosion : int
            Number of erosions applied to the TotalSegmentator skull before using it for refinement.

        Files Created
        -------------
        - head{file_number}.nii.gz : Original scan with non-head voxels set to -1000 HU (needed for registration).
        - mask{file_number}.nii.gz : Integer-labeled volume with values 0–4 indicating tissue types.

        Notes
        -----
        - Refined skull regions are derived from the intersection of soft tissue (head) and the
          eroded TotalSegmentator skull; they are given a distinct label to distinguish them from
          the original skull.
        - The final skull mask (`self.skull`) is updated to include the refined skull regions (kept binary).
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
        mask_soft_tissues = np.where(total_mask == 1, 1, 0) # Keeps only soft tissues (no arteries included)
        mask_eroded_skull = binary_erosion(self.skull_totalsegmentator, structure=generate_binary_structure(3,1), iterations=iter_erosion)
        not_included_skull = 3*mask_soft_tissues*mask_eroded_skull
        self.skull = self.skull + (mask_soft_tissues*mask_eroded_skull)
        total_mask = total_mask + not_included_skull

        # NIfTI file of the mask
        masked_image = nib.Nifti1Image(total_mask, self.img.affine, self.img.header) # Create a new NIfTI image
        nifti_path = os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii.gz")
        nib.save(masked_image, nifti_path)
        print(f"NIfTI generated : {nifti_path}")
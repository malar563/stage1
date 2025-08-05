import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import ants
import pandas as pd


class Identification:

    def __init__(self, big_output_directory="processed_files", file_number=0, fixed_img_path='icbm_avg_152_t1_tal_lin.nii', register_with_CT_not_normalized = False, show_normalized_pts=False):
        """
        Initialize the Identification class for landmark localization and image registration.

        This constructor sets up the registration environment by loading the fixed (atlas) and 
        moving (patient) images, initializing landmark coordinates, and reorienting the input 
        volumes for consistent processing. It also handles CT or MRI workflows depending on the 
        input flags.

        Parameters
        ----------
        big_output_directory : str, optional
            Root directory containing subfolders for each scan (default is "processed_files").

        file_number : int
            Identifier used to locate the directory and specific NIfTI and mask files.

        fixed_img_path : str
            Path to the fixed (atlas) image used for registration.

        register_with_CT_not_normalized : bool, optional
            If True, use CT (non-normalized) coordinates instead of MRI. Adjusts landmarks and file names accordingly.

        show_normalized_pts : bool, optional
            If True, display the predefined landmark positions over the fixed image for visual verification.

        Attributes
        ----------
        big_output_directory : str
            Base directory containing all output folders.

        file_number : str
            The folder name corresponding to the selected scan (casted from int).

        nifti_output_directory : str
            Full path to the subfolder containing the scan’s NIfTI files.

        moving_img_path : str
            Full path to the input (moving) image (head NIfTI file).

        fixed_img_path : str
            Full path to the reference (atlas) image.

        moving_img : ants.ANTsImage
            Reoriented moving image for registration.

        fixed_img : ants.ANTsImage
            Reoriented fixed (reference) image.

        initial_moving_img_orientation : str
            Orientation string of the input image before reorientation.

        orient_for_registration : str
            Target orientation for registration ("IAL", fixed for consistency across scans).

        nas_vox_normal_space, lpa_vox_normal_space, rpa_vox_normal_space : np.ndarray
            Hardcoded voxel coordinates of the anatomical landmarks in the normalized space.

        head : np.ndarray
            Binary mask volume representing the head, reoriented to (Z, X, Y) convention.

        filled_head : np.ndarray
            Copy of `head`, possibly post-processed (e.g., filled slices).

        moving_img_dimension : tuple
            Original dimensions of the loaded head mask volume.

        fwd_name, inv_name : str
            Names used for saving forward and inverse transformation fields, depending on CT/MRI.

        filled_y_slices : list
            Placeholder for storing filled slice indices (deprecated or not actively used).


        Future attributes
        -----------------
        Attributes you can access at one point in the identification process :
            -> fwd_df_transform, fwd_a_transform, inv_a_transform, inv_df_transform 
            -> registered_nasion, registered_lpa, registered_rpa, nasion, lpa, rpa :
            -> arteries

        Notes
        -----
        IMPORTANT :
            - For array manipulation and display purposes, in this class, when displaying an array, 
              note that the index order is (Z, X, Y) and NOT (Y, X, Z) as it was in the Segmentation class.
            - If you need to reorient an object :
                -> For a mask, use ants.reorient_image2(image, orientation) or reorient_to_original_masks
                -> For a point, use a affine matrix as found in the reorient_to_original_masks function
        """
        
        self.big_output_directory = big_output_directory
        self.file_number = str(file_number)
        self.nifti_output_directory = os.path.join(self.big_output_directory, self.file_number)
        self.moving_img_path = os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz")
        self.fixed_img_path = fixed_img_path

        self.register_with_CT_not_normalized = register_with_CT_not_normalized

        self.initial_moving_img = ants.image_read(self.moving_img_path) # Image provided by the mask
        self.initial_moving_img_orientation = ants.get_orientation(self.initial_moving_img) # Orientation of the mask
        self.orient_for_registration = "IAL" # Chosen orientation (easier to visualize). Do not change it : all the following code rely on this
        self.moving_img = ants.image_read(self.moving_img_path, reorient=self.orient_for_registration) # Image to be registered
        self.fixed_img = ants.image_read(fixed_img_path, reorient=self.orient_for_registration) # Reference image for registration

        # Landmark coordinates determined visually in the normalized space (mri)
        self.lpa_vox_normal_space = np.array([24, 112, 8]) # z, x, y
        self.rpa_vox_normal_space = np.array([24, 112, 172]) # z, x, y
        self.nas_vox_normal_space = np.array([28, 4, 90]) # z, x, y
        if not register_with_CT_not_normalized:
            if show_normalized_pts:
                self.show_3D_array(self.fixed_img.numpy(), axis=2, pts=[((self.nas_vox_normal_space[1],self.nas_vox_normal_space[0]), self.nas_vox_normal_space[2], "green"),
                                                                            ((self.lpa_vox_normal_space[1],self.lpa_vox_normal_space[0]), self.lpa_vox_normal_space[2], "blue"),
                                                                            ((self.rpa_vox_normal_space[1],self.rpa_vox_normal_space[0]), self.rpa_vox_normal_space[2], "red")])
                self.show_3D_array(self.fixed_img.numpy(), axis=1, pts=[((self.nas_vox_normal_space[2],self.nas_vox_normal_space[0]), self.nas_vox_normal_space[1], "green"),
                                                                            ((self.lpa_vox_normal_space[2],self.lpa_vox_normal_space[0]), self.lpa_vox_normal_space[1], "blue"),
                                                                            ((self.rpa_vox_normal_space[2],self.rpa_vox_normal_space[0]), self.rpa_vox_normal_space[1], "red")])
        self.fwd_name = "mri_fwd"+self.file_number
        self.inv_name = "mri_inv"+self.file_number
        
        # ------------------------ USER -----------------------------
        # Change the points of the CT here
        # from cropped_6_cow_angio__06__hv36__3
        if self.register_with_CT_not_normalized:
            # self.show_3D_array(self.moving_img.numpy(), axis=2)
            self.nas_vox_normal_space = np.array([101, 50, 238]) # z, x, y
            self.lpa_vox_normal_space = np.array([57, 236, 83]) # z, x, y
            self.rpa_vox_normal_space = np.array([69, 236, 403]) # z, x, y
        # -----------------------------------------------------------    
            if show_normalized_pts:
                self.show_3D_array(self.fixed_img.numpy(), axis=2, pts=[((self.nas_vox_normal_space[1],self.nas_vox_normal_space[0]), self.nas_vox_normal_space[2], "green"),
                                                                        ((self.lpa_vox_normal_space[1],self.lpa_vox_normal_space[0]), self.lpa_vox_normal_space[2], "blue"),
                                                                        ((self.rpa_vox_normal_space[1],self.rpa_vox_normal_space[0]), self.rpa_vox_normal_space[2], "red")])
                self.show_3D_array(self.fixed_img.numpy(), axis=1, pts=[((self.nas_vox_normal_space[2],self.nas_vox_normal_space[0]), self.nas_vox_normal_space[1], "green"),
                                                                        ((self.lpa_vox_normal_space[2],self.lpa_vox_normal_space[0]), self.lpa_vox_normal_space[1], "blue"),
                                                                        ((self.rpa_vox_normal_space[2],self.rpa_vox_normal_space[0]), self.rpa_vox_normal_space[1], "red")])  
            self.fwd_name = "ct_fwd"+self.file_number
            self.inv_name = "ct_inv"+self.file_number

        self.filled_y_slices = [] # Not used anymore : useful to know which y slices can be binary filled
        mask_img = nib.load(os.path.join(self.nifti_output_directory, "mask"+self.file_number+".nii.gz"))
        self.moving_img_dimension = mask_img.shape
        self.head = mask_img.get_fdata() 
        # IMPORTANT : from now on, it is [z,x,y] and not [y,x,z] as it was in the Segmentation class
        self.filled_head = self.reorient_to_original_masks(object_to_reorient=np.where(self.head>=1, 1, 0), 
                                                                initial_orient=self.initial_moving_img_orientation, 
                                                                final_orient=self.orient_for_registration)
        self.head = np.copy(self.filled_head)

   
    def show_3D_array(self, arr, axis=0, pts=None):
        """
        Display a 3D image or volume one slice at a time using a slider.

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
        - Points only appear on their corresponding slice.
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

    
    def reorient_to_original_masks(self, object_to_reorient=None, initial_orient="IAL", final_orient="RPI"):
        """
        Reorients a 3D volume or a point from one orientation code to another,
        by applying flips and axis transpositions based on direction labels.

        Parameters
        ----------
        object_to_reorient : np.ndarray, list, or tuple
            The object to reorient. Can be either:
            - A 3D NumPy array (volume),
            - A 3-element tuple/list/array representing a 3D point.

        initial_orient : str
            Orientation code of the input data (e.g., "IAL" = Inferior-Anterior-Left).

        final_orient : str
            Target orientation code to reorient to (e.g., "RPI" = Right-Posterior-Inferior).

        Returns
        -------
        object_reoriented : same type as input
            - If input is a 3D array, returns the reoriented volume.
            - If input is a 3D point, returns the coordinates in the new orientation.
            - If input is invalid, returns None.

        Side Effects
        ------------
        self.switch_lpa_rpa : bool
            Set to True if reorientation involves a Left-Right flip. Used for landmark relabeling.

        Notes
        -----
        - Orientation codes use anatomical directions:
            'R' = Right, 'L' = Left, 'A' = Anterior, 'P' = Posterior, 'I' = Inferior, 'S' = Superior.
        - A flip is applied if a direction in the `initial_orient` needs to be reversed to match `final_orient` (e.g., 'S'->'I').
        - Axes are reordered (transposed) if orientation letters are on different axes.
        """
        # Initialization
        initial_orient = list(initial_orient)
        final_orient = list(final_orient)
        dict_orientation = {"R":"L", "L":"R", "A":"P", "P":"A", "I":"S", "S":"I"}
        transpose_needed = []
        already_in = ["N", "N", "N"]
        self.switch_lpa_rpa = False

        # Determining transforms to apply
        for initial_axis, letter in enumerate(initial_orient):
            try:
                already_in[initial_axis]= (initial_axis, final_orient.index(letter)) # If already in, (initial axis number, final axis number)
            except:
                transpose_needed.append((letter, initial_axis)) # Initial axis where the letter needs to be changed
                new_letter = dict_orientation[letter]
                if new_letter == "L" or new_letter == "R":
                    self.switch_lpa_rpa = True
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
        elif isinstance(object_to_reorient, (np.ndarray, list, tuple)):
            # Checking if the dimensions are fitting with the orientation
            if self.initial_moving_img_orientation != initial_orient:
                initial_dimensions = self.moving_img_dimension[already_in[0][1]], self.moving_img_dimension[already_in[1][1]], self.moving_img_dimension[already_in[2][1]]
            else:
                initial_dimensions = self.moving_img_dimension

            initial_point = object_to_reorient[0], object_to_reorient[1], object_to_reorient[2], 1
            affine = np.zeros((4,4))
            # Place ones at the right place to transpose
            affine[0,already_in[0][1]] = 1
            affine[1,already_in[1][1]] = 1
            affine[2,already_in[2][1]] = 1
            affine[3,3] = 1
            # Flip axis
            for letter, initial_index in transpose_needed:
                new_index = already_in[initial_index][1]
                affine[new_index] = -1*affine[new_index]
                affine[new_index,-1] = initial_dimensions[initial_index]-1
            # print(affine)
            new_point = (affine @ initial_point)[:-1]
            return new_point
        
        # Dealing with exeptions
        return None    


    def register(self, show=False):
        """
        Perform non-linear registration (SyN) between the moving (patient) and fixed (atlas) images.

        Uses ANTsPy to compute affine and deformable transformations (SyN ; Symmetric Normalization). 
        Saves the transformation files (forward and inverse) in the output directory, and stores them 
        as class attributes for later use (e.g., point or image transformation).

        Parameters
        ----------
        show : bool, optional (default=False)
            If True, displays 3D views of:
            - The original moving image
            - The original fixed image
            - The warped moving image (after registration)
            - The warped fixed image

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
        shutil.copy(transformation['fwdtransforms'][0], os.path.join(self.nifti_output_directory, self.fwd_name+".nii.gz"))
        ants.write_transform(transform=ants.read_transform(transformation['fwdtransforms'][1]), filename=os.path.join(self.nifti_output_directory, self.fwd_name+".mat"))
        ants.write_transform(transform=ants.invert_ants_transform(ants.read_transform(transformation['invtransforms'][0])), filename=os.path.join(self.nifti_output_directory, self.inv_name+".mat"))
        shutil.copy(transformation['invtransforms'][1], os.path.join(self.nifti_output_directory, self.inv_name+".nii.gz"))

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
        self.fwd_df_transform = ants.read_transform(os.path.join(self.nifti_output_directory, self.fwd_name+".nii.gz")) # .nii.gz -> deformation field (df) fwd_df_transform != inv_df_transform
        self.fwd_a_transform = ants.read_transform(os.path.join(self.nifti_output_directory, self.fwd_name+".mat")) # .mat -> affine transform (a) fwd_a_transform = inv_a_transform 
        self.inv_a_transform = ants.read_transform(os.path.join(self.nifti_output_directory, self.inv_name+".mat")) # .mat -> affine transform (a) 
        self.inv_df_transform = ants.read_transform(os.path.join(self.nifti_output_directory, self.inv_name+".nii.gz")) # .nii.gz -> deformation field (df)


    def find_registered_lpa_rpa_nasion(self):
        """
        Compute patient-space coordinates of anatomical landmarks (LPA, RPA, Nasion)
        by transforming normalized-space voxel indices using ANTs forward transforms.

        The method uses known voxel positions in the atlas (normalized space), converts them to
        physical coordinates, applies the forward deformation and affine transforms, and 
        maps the resulting physical points back to voxel indices in the patient image.

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
        - Applies `fwd_df_transform` followed by `fwd_a_transform` to reach patient space.
        - Final physical points are mapped to voxel space using the patient image.
        - Coordinates are rounded to integer values (z, x, y) to get the position in the array.   
        """
        # LPA : automatically identify on patient
        lpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.lpa_vox_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, lpa_pt_normal_space)
        lpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, lpa_pt_patient_space)
        lpa_vox_patient_space =  ants.transform_physical_point_to_index(self.moving_img, lpa_pt_patient_space)
        self.registered_lpa = round(lpa_vox_patient_space[0]), round(lpa_vox_patient_space[1]), round(lpa_vox_patient_space[2])
        # print('registered LPA', self.registered_lpa) # (z,x,y)

        # RPA : automatically identify on patient
        rpa_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.rpa_vox_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, rpa_pt_normal_space)
        rpa_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, rpa_pt_patient_space)
        rpa_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, rpa_pt_patient_space)
        self.registered_rpa = round(rpa_vox_patient_space[0]), round(rpa_vox_patient_space[1]), round(rpa_vox_patient_space[2])
        # print('registered RPA', self.registered_rpa) # (z,x,y)

        # nasion : automatically identify on patient
        nas_pt_normal_space = ants.transform_index_to_physical_point(self.fixed_img, self.nas_vox_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_df_transform, nas_pt_normal_space)
        nas_pt_patient_space = ants.apply_ants_transform_to_point(self.fwd_a_transform, nas_pt_patient_space)
        nas_vox_patient_space = ants.transform_physical_point_to_index(self.moving_img, nas_pt_patient_space)
        self.registered_nasion = round(nas_vox_patient_space[0]), round(nas_vox_patient_space[1]), round(nas_vox_patient_space[2])
        # print('registered nasion', self.registered_nasion) # (z,x,y)


    def _fill_cavities(self):
        """
        Fill internal cavities in the binary head mask along all three anatomical planes.

        This method applies 2D hole-filling (using `scipy.ndimage.binary_fill_holes`) 
        to each slice of the `self.filled_head` mask independently along the axial (xy), 
        coronal (yz), and sagittal (xz) planes.

        Updates
        -------
        self.filled_head : np.ndarray
            Binary head mask after cavity filling.
        self.filled_y_slices : list[int]
            Indices of sagittal slices (xz-plane) where cavity filling occurred.

        Note
        ----
        - Will be called by the function find_nasion
        """        
        from scipy.ndimage import binary_fill_holes

        for i in range(0, len(self.filled_head[1,1,:])):
            # Fill holes in the x-z plane
            original = self.filled_head[:,:,i]
            filled = binary_fill_holes(self.filled_head[:,:,i])
            if not np.array_equal(original,filled): # Check if the slice has been filled
                self.filled_y_slices.append(i)
            self.filled_head[:,:,i] = filled
        for i in range(0, len(self.filled_head[1,:,1])):
            # Fill holes in the y-z plane 
            self.filled_head[:,i,:] = binary_fill_holes(self.filled_head[:,i,:]) 
        for i in range(0, len(self.filled_head[:,1,1])):
            # Fill holes in the x-y plane
            self.filled_head[i,:,:] = binary_fill_holes(self.filled_head[i,:,:])
    

    def find_nasion(self, window=15):
        """
        Refine the position of the nasion using local density analysis on the binary head mask.

        Starting from the registered (approximate) nasion location, this method examines a local 
        3D region (Region of Interest) - the patient's face - in the filled head mask to find a more anatomically accurate 
        voxel by analyzing binary mask density along the x-axis.

        Parameters
        ----------
        window : int, optional
            Half-width of the cubic region of interest (ROI) centered around the initial nasion 
            estimate. The full ROI has shape (2*window, 2*window, 2*window). Default is 15.

        Sets
        ----
        self.nasion : tuple[int, int, int]
            Refined voxel coordinates of the nasion in (z, x, y) format.

        Notes
        -----
        - Internally calls `fill_cavities()` to ensure the binary mask is representative of the person's face before analysis.
        - Optimizes the location of the nasion by maximizing the density along the y-axis and minimizing it along the z-axis.
        """
        self._fill_cavities()

        reg_nas_z, reg_nas_x, reg_nas_y = self.registered_nasion
        
        # Limits to avoid a too big window
        z_min, z_max = 0, self.filled_head.shape[0]
        x_min, x_max = 0, self.filled_head.shape[1]
        y_min, y_max = 0, self.filled_head.shape[2]

        # Secure slicing
        z_start = max(z_min, reg_nas_z - window)
        z_end   = min(z_max, reg_nas_z + window)
        x_start = max(x_min, reg_nas_x - window)
        x_end   = min(x_max, reg_nas_x + window)
        y_start = max(y_min, reg_nas_y - window)
        y_end   = min(y_max, reg_nas_y + window)

        ROI_nas = self.filled_head[z_start:z_end, x_start:x_end, y_start:y_end]
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(ROI_nas, axis = 1) # axis=2 sums in y, axis=0 sums in z
        # To see the person's face, uncomment this:
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

        if np.any(nasion_row == 0) or np.any(nasion_row == (window*2) - 1):
            nasion_y_final = reg_nas_y
        if np.any(nasion_column == 0) or np.any(nasion_column == (window*2) - 1):
            nasion_z_final = reg_nas_z

        nasion_x_final = np.nonzero(self.filled_head[nasion_z_final,:,nasion_y_final])[0][0]
        self.nasion = nasion_z_final, nasion_x_final, nasion_y_final # (z, x, y)


    def check_nasion(self):
        """
        Used to visually validate the refined nasion position against the initially registered one.

        This function plots axial, coronal, and sagittal views of the head mask, overlaying
        both the refined nasion (in red) and the registered nasion (in blue) to assess the
        accuracy of the correction.
        """
        # x-axis goes through the nasion
        # Axial view : fixed z-value
        plt.imshow(self.head[self.nasion[0],:,:], origin="lower")
        plt.scatter([self.nasion[2]], [self.nasion[1]], c="r")
        plt.scatter([self.registered_nasion[2]], [self.registered_nasion[1]], c="b")
        plt.show()
        # Coronal view : fixed x-value
        plt.imshow(self.head[:,self.nasion[1],:], origin="lower")
        plt.scatter([self.nasion[2]], [self.nasion[0]], c="r")
        plt.scatter([self.registered_nasion[2]], [self.registered_nasion[0]], c="b")
        plt.show()
        # Sagittal view : fixed y-value
        plt.imshow(self.head[:,:,self.nasion[2]], origin="lower")
        plt.scatter([self.nasion[1]], [self.nasion[0]], c="r")
        plt.scatter([self.registered_nasion[1]], [self.registered_nasion[0]], c="b")
        plt.show()
 

    def _find_depth_rpa(self):
        """
        Find the surface (y value) of the binary volume at a fixed x,z location

        Returns:
            int: y-index corresponding to the surface of the head at the RPA x,z location.
        """
        nonzero = np.nonzero(self.filled_head[self.registered_rpa[0],self.registered_rpa[1],:])[0]
        if self.registered_rpa[2] in nonzero:   
            index_rpa_y = np.where(nonzero == self.registered_rpa[2])[0][0]
            for i in range(1, len(nonzero)-index_rpa_y):
                if nonzero[index_rpa_y+i] != self.registered_rpa[2]+i:
                    return nonzero[index_rpa_y+i-1] # gets the surface of the head
            return nonzero[-1] # gets the surface of the head
        else:
            index_lpa = np.argmin(np.abs(nonzero-self.registered_rpa[2]))
            return nonzero[index_lpa] # gets the surface of the head
        


    def _find_depth_lpa(self):
        """
        Find the surface (y value) of the binary volume at a fixed x,z location

        Returns:
            int: y-index corresponding to the surface of the head at the RPA x,z location.
        """
        nonzero = np.nonzero(self.filled_head[self.registered_lpa[0],self.registered_lpa[1],:])[0]
        if self.registered_lpa[2] in nonzero:  
            index_lpa_y = np.where(nonzero == self.registered_lpa[2])[0][0]
            for i in range(1, index_lpa_y):
                if nonzero[index_lpa_y-i] != self.registered_lpa[2]-i:
                    return nonzero[index_lpa_y-i+1] # gets the surface of the head
            return nonzero[0] # gets the surface of the head
        else:
            index_lpa = np.argmin(np.abs(nonzero-self.registered_lpa[2]))
            return nonzero[index_lpa] # gets the surface of the head
        
    
    def improve_lpa_rpa(self):
        """
        Adjust the LPA and RPA y value to align them with the head surface.

        Note
        ----
        - Calls _find_depth_rpa and _find_depth_lpa 
        """
        self.lpa = self.registered_lpa[0], self.registered_lpa[1], self._find_depth_lpa()
        self.rpa = self.registered_rpa[0], self.registered_rpa[1], self._find_depth_rpa()


    def save_pts_to_csv(self):
        """
        Save MRI or CT anatomical landmarks (Nasion, LPA, RPA) in voxel space to a CSV file.

        This function updates the points CSV file (`points{file_number}.csv`) by inserting
        refined and registered coordinates for both MRI and CT, depending on registration type.

        Notes
        -----
        - The coordinates are reordered as (X, Y, Z) for readability and compatibility.
        - Automatically switches LPA and RPA values if required.
        - Overwrites the target CSV file with updated contents.
        - Preserves existing data before and after the landmarks section.
        - Reorients coordinates from registration space back to the original orientation 
          (orientation of the mask just after the segmentation process).

        """
        csv_path = os.path.join(self.nifti_output_directory, f"points{self.file_number}.csv")

        df_existing = pd.read_csv(csv_path)
        df_to_keep = df_existing.values[:5,:4].tolist()
        df_end = df_existing.values[17:,:4].tolist()
        df_is_exist = np.array(df_existing.values[5:,:4].tolist())

        improved_landmarks = [self.nasion, self.lpa, self.rpa]
        registered_landmarks = [self.registered_nasion, self.registered_lpa, self.registered_rpa]
        for i, point in enumerate(improved_landmarks):
            improved_landmarks[i] = self.reorient_to_original_masks(point,
                                                              initial_orient=self.orient_for_registration,
                                                              final_orient=self.initial_moving_img_orientation)
        for i, point in enumerate(registered_landmarks):
            registered_landmarks[i] = self.reorient_to_original_masks(point,
                                                              initial_orient=self.orient_for_registration,
                                                              final_orient=self.initial_moving_img_orientation)
        if self.register_with_CT_not_normalized:
            mri_improved_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]
            mri_registered_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]
            ct_improved_landmarks = improved_landmarks
            ct_registered_landmarks = registered_landmarks
        else:
            mri_improved_landmarks = improved_landmarks
            mri_registered_landmarks = registered_landmarks
            ct_improved_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]
            ct_registered_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]

        data = np.array([
                    ["MRI Nasion improved (voxel)", mri_improved_landmarks[0][1], mri_improved_landmarks[0][0], mri_improved_landmarks[0][2]],
                    ["MRI Nasion registered (voxel)", mri_registered_landmarks[0][1], mri_registered_landmarks[0][0], mri_registered_landmarks[0][2]],
                    ["MRI LPA improved (voxel)", mri_improved_landmarks[1][1], mri_improved_landmarks[1][0], mri_improved_landmarks[1][2]],
                    ["MRI LPA registered (voxel)", mri_registered_landmarks[1][1], mri_registered_landmarks[1][0], mri_registered_landmarks[1][2]],
                    ["MRI RPA improved (voxel)", mri_improved_landmarks[2][1], mri_improved_landmarks[2][0], mri_improved_landmarks[2][2]],
                    ["MRI RPA registered (voxel)", mri_registered_landmarks[2][1], mri_registered_landmarks[2][0], mri_registered_landmarks[2][2]],
                    ["CT Nasion improved (voxel)", ct_improved_landmarks[0][1], ct_improved_landmarks[0][0], ct_improved_landmarks[0][2]],
                    ["CT Nasion registered (voxel)", ct_registered_landmarks[0][1], ct_registered_landmarks[0][0], ct_registered_landmarks[0][2]],
                    ["CT LPA improved (voxel)", ct_improved_landmarks[1][1], ct_improved_landmarks[1][0], ct_improved_landmarks[1][2]],
                    ["CT LPA registered (voxel)", ct_registered_landmarks[1][1], ct_registered_landmarks[1][0], ct_registered_landmarks[1][2]],
                    ["CT RPA improved (voxel)", ct_improved_landmarks[2][1], ct_improved_landmarks[2][0], ct_improved_landmarks[2][2]],
                    ["CT RPA registered (voxel)", ct_registered_landmarks[2][1], ct_registered_landmarks[2][0], ct_registered_landmarks[2][2]]])
        
        if len(df_is_exist) != 0:
            if self.register_with_CT_not_normalized:
                data[0:6,:] = df_is_exist[0:6,:]
            else:
                data[6:12,:] = df_is_exist[6:12,:]

        if self.switch_lpa_rpa: # Switches LPA and RPA if this axis was flipped during the identification process
            if self.register_with_CT_not_normalized:
                new_lpa = data[10:12,1:].copy()
                new_rpa = data[8:10,1:].copy()
                data[10:12,1:] = new_rpa
                data[8:10,1:] = new_lpa
            else:
                new_lpa = data[4:6,1:].copy()
                new_rpa = data[2:4,1:].copy()
                data[4:6,1:] = new_rpa
                data[2:4,1:] = new_lpa

        data = data.tolist()
        df_to_keep += data
        df_to_keep += df_end
        df = pd.DataFrame(df_to_keep)
        df.to_csv(csv_path, index=False)


    def delete_useless_files(self):
        """
        Remove unnecessary NIfTI files from the output directory to save space.

        Files deleted:
        - Forward and inverse transform files (MAT and NIfTI)
        - The output of TotalSegmentator
        - The original (uncropped) CT scan if a cropped version exists

        This helps clean up the folder after segmentation and registration steps.

        Notes
        -----
        - Files that don’t exist will simply be skipped with a message.
        - It’s safe to run even if some files are already missing.
        - IMPORTANT: Call this only after registration is complete.
        """
        # Listing NIfTI files in the folder
        useless_files = [os.path.join(self.nifti_output_directory, self.fwd_name+".mat"),
                         os.path.join(self.nifti_output_directory, self.fwd_name+".nii.gz"),
                         os.path.join(self.nifti_output_directory, self.inv_name+".mat"),
                         os.path.join(self.nifti_output_directory, self.inv_name+".nii.gz"),
                         os.path.join(self.nifti_output_directory, "totalsegmentator"+self.file_number+".nii.gz")] 
        
        # Finds if a cropped file exists, and delete the original one in that case
        nii_files = [f for f in os.listdir(self.nifti_output_directory) if f.endswith(".nii") or f.endswith(".nii.gz")]
        cropped_files = [f for f in nii_files if f.startswith("cropped_")]
        if len(cropped_files)>=1:
            excluded_prefixes = ("cropped_", "ct_fwd", "mri_fwd", "ct_inv", "mri_inv", "mask", "mca_territory", "totalsegmentator", "head") 
            non_cropped = [f for f in nii_files if not f.startswith(excluded_prefixes)]
            if non_cropped:
                non_cropped = non_cropped[0]
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
        -----
        self.arteries : ants.ANTsImage
            Binary mask of segmented arteries within the MCA territory in patient space.

        Output
        ------
        - A NIfTI file named "mca_territory{self.file_number}.nii.gz" is saved to the output directory.

        Notes
        -----
        - This function only runs if MRI normalization is used (`register_with_CT_not_normalized` is False).
        - Includes visualization steps for intermediate masks and projections (you will need to uncomment it).
        """
        if not self.register_with_CT_not_normalized:
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
            self.arteries = self.arteries.new_image_like(arteries_only)
            # self.show_3D_array(self.arteries.numpy())

            # Bringing the arteries mask in the normalized space 
            registered_arteries = ants.apply_transforms(fixed=resampled_mca_territories, moving=self.arteries, transformlist=[os.path.join(self.nifti_output_directory, "inv"+self.file_number+".mat"), os.path.join(self.nifti_output_directory, "inv"+self.file_number+".nii.gz")])
            # self.show_3D_array(registered_arteries.numpy())

            # Keeping the arteries of the mask included in the MC territories
            normalized_mca_arteries = registered_arteries.numpy()*resampled_mca_territories.numpy()
            # self.show_3D_array(normalized_mca_arteries)

            # Bringing the MCA arteries back to the patient's space
            normalized_mca_arteries = registered_arteries.new_image_like(normalized_mca_arteries)
            patient_mca_arteries = ants.apply_transforms(fixed=self.arteries, moving=normalized_mca_arteries, transformlist=[os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".mat"), os.path.join(self.nifti_output_directory, "fwd"+self.file_number+".nii.gz")])
            
            # self.show_3D_array(patient_mca_arteries.numpy())
            counts_z = np.sum(patient_mca_arteries.numpy(), axis = 0) # axis=2 sums in y, axis=0 sums in z
            # plt.imshow(counts_z, origin="lower")
            # plt.show()

            # Nifti file with the MCA territories
            head_img =  nib.load(os.path.join(self.nifti_output_directory, "head"+self.file_number+".nii.gz"))
            patient_mca_arteries = ants.reorient_image2(patient_mca_arteries, orientation=self.initial_moving_img_orientation) # Reorient accordinf to the initial mask
            mca_image = nib.Nifti1Image(patient_mca_arteries.numpy(), head_img.affine, head_img.header) # Create a new NIfTI image
            nifti_path = os.path.join(self.nifti_output_directory, "mca_territory"+self.file_number+".nii.gz")
            nib.save(mca_image, nifti_path)
            print(f"NIfTI generated : {nifti_path}.nii.gz")

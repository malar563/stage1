import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label


a = np.array([[[0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [1,1,0,0,1,0],
              [0,0,0,1,1,0]], 
              [[0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [1,1,0,0,1,0],
              [0,0,0,1,1,0]]])
labeled_array, num_features = label(a)
# print(labeled_array, num_features)


x = np.bincount(np.array([1, 1, 1, 1, 1]))
# print(x)

y = np.bincount(np.flip(np.array([0, 1, 1, 3, 2, 1, 7])))
# print(y)

indices = np.where(a)
# print(indices)

from scipy.ndimage import binary_fill_holes, binary_closing
b = np.array([[[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 0, 1, 1],
       [1, 1, 0, 1, 1],
       [1, 1, 1, 1, 1]], 
       [[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]], 
       [[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]],
       [[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]], 
       [[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]]])
# print(binary_fill_holes(b))
# print(binary_closing(b))

from scipy.ndimage import generate_binary_structure, binary_dilation
from skimage.morphology import ball
balle = ball(2)
struct = generate_binary_structure(3, 1)
# print(balle)
# print(struct)


n = 5
array_3d = np.zeros((n, n, n)) 
array_3d[n//2,n//2,n//2] = 2
# print(array_3d)


c = np.array([[1,2,3],[4,5,6],[7,8,9]])
c_flat = np.ravel(c)
# print(c_flat)

# print(binary_dilation(array_3d, generate_binary_structure(3,3)))
# print(binary_dilation(array_3d, generate_binary_structure(3,3), iterations=2))

import scipy.ndimage as ndimage

A = np.array([[[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 2, 2, 0, 0, 0],
              [0, 1, 1, 2, 2, 0, 3, 3],
              [0, 0, 0, 0, 0, 0, 3, 3],
              [0, 0, 0, 0, 0, 0, 3, 3],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 4, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],[[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],[[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]])

B = ndimage.maximum_filter(A, 3)
B[A != 0] = A[A != 0]

# print(B)


from scipy.ndimage import distance_transform_edt

binary = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0]
], dtype=bool)
# print(binary)
# print(~binary)
binary = binary != 1
# print(binary)

distance = distance_transform_edt(~binary)
# print(distance)

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

nose = np.array([214, 214, 214, 214, 214, 214, 214, 216, 226, 230, 233, 235, 237, 239, 241, 242, 244, 245, 246, 247, 247, 248, 248, 249, 249, 250, 250, 250, 250, 250, 251, 251, 251, 250, 250, 250, 249, 249, 249, 248, 248, 247, 247, 247, 246, 246, 245, 244, 244, 243, 243, 242, 241, 240, 239, 239, 238, 237, 236, 235, 234, 233, 233, 232, 231, 231, 235, 233, 227, 226, 226, 225, 224, 223, 222, 222, 221, 220, 219, 219, 218, 217, 216, 215, 214, 214, 213, 212, 211, 211, 210, 209, 209, 208, 208, 207, 207, 206, 206, 205, 204, 204, 205, 205, 206, 206, 206, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 210, 211, 211, 211, 211, 211, 211, 211, 211, 212, 212, 212, 212, 212, 212, 212, 212, 212, 212, 213, 213, 213, 213, 213, 213, 212, 212, 212, 212, 212, 211, 211, 210, 210, 210, 210, 209, 209, 208, 208, 208, 207, 207, 207, 207, 206, 206, 206, 206, 206, 205, 205, 205, 205, 204, 204, 204, 204, 204, 203, 203, 203, 202, 202, 201, 200, 199, 199, 199, 199, 199, 198, 198, 197, 197, 196, 195, 195, 194, 193, 193, 192, 191, 191, 190, 189, 188, 188, 187, 186, 186, 185, 184, 183, 182, 181, 180, 180, 178, 178, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 165, 164, 163, 162, 161, 160, 158, 157, 155, 154, 152, 151, 149, 147, 146, 144, 142, 140, 138, 136, 133, 130, 127, 124, 120, 117, 114, 111])
# print(find_peaks(gaussian_filter1d(-1*nose, sigma=2)))
plt.plot(gaussian_filter1d(-1*nose, sigma=2))
# plt.show()

binary = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
# print(np.nonzero(binary)[0][0])

voxel_index = np.array([100, 225, 256])
# print(np.append(voxel_index, 1))

import ants

fwd_transform = np.array([ 9.73895967e-01, -3.47097553e-02,  3.70225497e-02,  4.83456627e-03, 8.98923755e-01, -5.44617735e-02, -3.11447047e-02, 1.33252731e-02, 9.55993474e-01,  6.73420477e+00, -2.17528336e+02, -7.79117371e+02])
inv_params = fwd_transform.reshape(3, 4)
# print(inv_params)
# print(inv_params[:,:-1], inv_params[:,-1])
sub_inv_params = np.linalg.inv(inv_params[:,:-1])
inv_params[:,-1] = np.matmul(-1*sub_inv_params, inv_params[:,-1])
inv_params[:,:-1] = sub_inv_params
# print(inv_params)
# print(inv_params.reshape(1,12)[0])

# transform = ANTsTransform()
# import nibabel as nib
# nifti_img = nib.load('nifti/2/brain2.nii')
# voxel_index = np.array([25, 28, 100, 1])
# print(nifti_img.affine)
# print(np.matmul(nifti_img.affine, voxel_index)) # Ne fonctionne pas car donne en RAS+ coordinate pour nibabel
  

arr = np.array([6, 7, 7, 8, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 16, 16,
                17, 17, 17, 17, 17, 16, 16, 15, 14, 13, 13, 12, 11, 11, 10, 9, 8, 8, 7, 7, 6])
# Find all indices where the value is 17
indices_17 = np.where(arr >= 14)[0]
# print(indices_17)
# Get the index of the middle one
middle_index = indices_17[len(indices_17) // 2]
# print("Index of the middle 17:", middle_index)
# print(len(arr))


txt = "Hello, welcome to my world."
x = txt.startswith("Hello")
# print(x)

txt = "welcome to the jungle"
x = txt.split()
# print(x)

# def reorient_point_to_original_mask(orient_init="RPI", orient_fin="IAL"):
def reorient_point_to_original_mask(orient_init="RPS", orient_fin="LPS"):
    orient_init = list(orient_init)
    orient_fin = list(orient_fin)

    dict_orientation = {"R":"L", "L":"R", "A":"P", "P":"A", "I":"S", "S":"I"}
    transpose_needed = []
    already_in = ["N", "N", "N"]
    for axis_init, letter in enumerate(orient_init):
        try:
            already_in[axis_init]= (axis_init, orient_fin.index(letter)) # If already in, (initial axis number, final axis number)
        except:
            transpose_needed.append((letter, axis_init)) # Initial axis where the letter needs to be changed
            new_letter = dict_orientation[letter]
            orient_init[axis_init] = new_letter
    for axis_init, element in enumerate(orient_init):
        already_in[axis_init] = (axis_init, orient_fin.index(element))

    print(orient_init, orient_fin)
    print(already_in, transpose_needed)

    affine = np.zeros((4,4))
    affine[0,already_in[0][1]] = 1
    affine[1,already_in[1][1]] = 1
    affine[2,already_in[2][1]] = 1
    affine[3,3] = 1
    resolution= 256, 512, 512
    for letter, initial_index in transpose_needed:
        new_index = already_in[initial_index][1]
        affine[new_index] = -1*affine[new_index]
        affine[new_index,-1] = resolution[initial_index]-1
    print(affine)

    initial_point = np.array([225, 44, 259, 1])
    initial_point = np.array([101, 50, 233, 1])
    new_point = (affine @ initial_point)[:-1]
    print(new_point)

    arr = np.array([[[ 0.,  0.,  0.,  0.],
          [ 0.,  1.,  0.,  0.],
          [ 0.,  0.,  0.,  0.]],
          [[ 0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  2.],
          [ 0.,  0.,  0.,  0.]],
          [[ 0.,  0.,  0.,  0.],
          [ 0.,  3.,  0.,  0.],
          [ 0.,  0.,  0.,  0.]]])
    arr = np.array([[ 0.,  0.,  0.,  0.],
          [ 0.,  1.,  0.,  0.],
          [ 0.,  0.,  0.,  0.]])
    print(isinstance(arr, np.ndarray), len(arr.shape))
    for element, axis_to_flip in transpose_needed:
        arr = np.flip(arr, axis=axis_to_flip)
    arr = np.transpose(arr, (already_in[0][1], already_in[1][1], already_in[2][1]))
    print(arr)

# reorient_point_to_original_mask()
def save_pts_to_csv():
    import pandas as pd
    import os    
    csv_path = "test.csv"

    # df_existing = pd.read_csv(csv_path)
    # print(list(df_existing.iloc[:,:]))

    # list_of_rows = df_existing.values
    # print(list_of_rows)

    data = [["","x", "y", "z"],
                ["Dimensions", 1, 1, 1],
                ["Resolution (mm)", 2, 2, 2],
                ["Length (mm)", 3, 3, 3]]
    df = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df.values[:,:4] = np.array(data)
    df.to_csv(csv_path, index=False, header=False)
    print(df.values)   
# save_pts_to_csv()

df_to_keep = ["asdnasjds", 5, (54,5)]
data = ["Nasion", 1, 2, 3],["LPA", 4, 5, 6],["RPA", 7, 8, 9]
df_to_keep += data
# print(df_to_keep)


# import nibabel as nib
# import os
# img = nib.load("jspakoi/1/6_cow_angio__06__hv36__3.nii.gz")
# img = nib.load("jspakoi/1/cropped_6_cow_angio__06__hv36__3.nii.gz")
# path="jspakoi/1/6_cow_angio__06__hv36__3.nii.gz"
# path="jspakoi/1/cropped_6_cow_angio__06__hv36__3.nii.gz"
# # print(img.header)
# # print(path.removeprefix("cropped_"))
# # print(len([f for f in os.listdir("nifti/2") if f.startswith('cropped')]))
# # print(img.header["dim"][1:4])

# # Lister les fichiers NIfTI dans le dossier
# nii_directory = "cava/0"
# nii_directory = "nifti/2"
# # nii_directory = "jspakoi/1"
# nii_files = [f for f in os.listdir(nii_directory) if f.endswith(".nii.gz") or f.endswith(".nii")]
# print(nii_files)

# # Chercher un fichier qui commence par "cropped_"
# cropped_files = [f for f in nii_files if f.startswith("cropped_")]
# print(cropped_files)

# if cropped_files:
#     # Si un fichier "cropped_" est trouvé, on l’utilise
#     nii_path = os.path.join(nii_directory, cropped_files[0])
#     not_cropped_nii_path = cropped_files[0].removeprefix("cropped_")
#     print(not_cropped_nii_path)
# else:
#     # Sinon, chercher le fichier sans "cropped_"
#     all_non_cropped = [f for f in nii_files if not f.startswith("cropped_")]
#     print(all_non_cropped)
#     if all_non_cropped:
#         nii_path = os.path.join(nii_directory, all_non_cropped[0])
#         not_cropped_nii_path = os.path.basename(nii_path)
#         print(not_cropped_nii_path)
#     # else:
#     #     # Si aucun NIfTI n’existe, on génère le fichier à partir du DICOM
#     #     print("No NIfTI file found. Processing the specified DICOM file...")
#     #     self.dcm_to_nii()
#     #     self.img = nib.load(self.nii_path)
#     #     self.array = self.img.get_fdata()
#     #     self.resolution = tuple(np.abs(self.img.affine[i][i]) for i in range(3))
#     #     self.dimension = self.img.shape
#     #     self.save_to_csv()
#     #     return

#     # Chargement du fichier trouvé
# img = nib.load(nii_path)
# array = img.get_fdata()
# resolution = tuple(np.abs(img.affine[i][i]) for i in range(3))
# dimension = img.shape
# # print(f"NIfTI found: {nii_path}")

# # save_to_csv()





x = np.array([0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,0])
nonzero = np.nonzero(x)[0]
# print(nonzero)
filled_y_slices = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
lpa_y = 20
rpa_y = 24


def find_depth_lpa():
    if lpa_y in nonzero:
        print(nonzero)   
        index_lpa_y = np.where(nonzero == lpa_y)[0][0]
        print(index_lpa_y)
        for i in range(1, index_lpa_y):
            if nonzero[index_lpa_y-i] != lpa_y-i:
                return nonzero[index_lpa_y-i+1] # gets the surface of the head
        return nonzero[0] # gets the surface of the head
    else:
        index_lpa = np.argmin(np.abs(nonzero-lpa_y))
        return nonzero[index_lpa] # gets the surface of the head
    
# print(find_depth_lpa())

def find_depth_rpa():
    if rpa_y in nonzero:
        print(nonzero)   
        index_rpa_y = np.where(nonzero == rpa_y)[0][0]
        print(index_rpa_y)
        for i in range(1, len(nonzero)-index_rpa_y):
            if nonzero[index_rpa_y+i] != rpa_y+i:
                return nonzero[index_rpa_y+i-1] # gets the surface of the head
        return nonzero[-1] # gets the surface of the head
    else:
        index_lpa = np.argmin(np.abs(nonzero-rpa_y))
        return nonzero[index_lpa] # gets the surface of the head
    
# print(find_depth_rpa())


import os
import re


def get_dicom_folders(directory="nifti", green_words = ["THIN", "thin"]):

    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    final_folders = []
    dict_resolution = {}

    for folder in folders:
        dcm_folders = []
        print("new dcm folder")
        for root, dirs, files in os.walk(os.path.join(directory,folder)):
            # print("Root:", root)
            # print("Directories:", dirs)
            # print("Files:", files)

            # Only takes folders with no subfolders
            if len(dirs) == 0:
                dcm_folders.append(root)

        # If there is only one file containing dicoms, directly takes it
        if len(dcm_folders) == 1:
            final_folders.append(dcm_folders[0])
        
        elif len(dcm_folders) > 1:
            for dcm_folder_path in dcm_folders:
                
                # If a green word is found in the name of the path, directly takes it
                if any(green_word in dcm_folder_path for green_word in green_words):
                    final_folders.append(dcm_folder_path)

                # Takes the file with the smallest number (resolution) mentionned
                basename = os.path.basename(dcm_folder_path)
                number_str = re.findall(r"[-+]?(?:\d*\.*\d+)", basename)
                number = [float(i) for i in number_str]
                if len(number) != 0:
                    dict_resolution[number[0]] = dcm_folder_path
            try:
                final_folders.append(dict_resolution[min(dict_resolution.keys())])
            except:
                pass # Can't put final_folders.append(None) here because it will appear if there is a green word too
        else:
            final_folders.append(None)

    return final_folders
                
# print(get_dicom_folders())

# if not False:
    # print("allo")


import pandas as pd
def save_pts_to_csv(register_with_CT_not_normalized=True):
    csv_path = "points0.csv"

    df_existing = pd.read_csv(csv_path)
    df_to_keep = df_existing.values[:5,:4].tolist()
    df_is_exist = np.array(df_existing.values[5:22,:4].tolist())
    
    if register_with_CT_not_normalized:
        mri_improved_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]
        mri_registered_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]
        ct_improved_landmarks = [(1,2,3), (4,5,6), (7,8,9)]
        ct_registered_landmarks = [(-1,-2,-3), (-4,-5,-6), (-7,-8,-9)]
    else:
        mri_improved_landmarks = [(1,2,3), (4,5,6), (7,8,9)]
        mri_registered_landmarks = [(-1,-2,-3), (-4,-5,-6), (-7,-8,-9)]
        ct_improved_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]
        ct_registered_landmarks = [("-","-","-"), ("-","-","-"), ("-","-","-")]

    data = np.array([["----------Landmarks found with normalized MRI----------","nan","nan","nan"],
                ["Nasion improved (voxel)", mri_improved_landmarks[0][1], mri_improved_landmarks[0][0], mri_improved_landmarks[0][2]],
                ["Nasion registered (voxel)", mri_registered_landmarks[0][1], mri_registered_landmarks[0][0], mri_registered_landmarks[0][2]],
                ["LPA improved (voxel)", mri_improved_landmarks[1][1], mri_improved_landmarks[1][0], mri_improved_landmarks[1][2]],
                ["LPA registered (voxel)", mri_registered_landmarks[1][1], mri_registered_landmarks[1][0], mri_registered_landmarks[1][2]],
                ["RPA improved (voxel)", mri_improved_landmarks[2][1], mri_improved_landmarks[2][0], mri_improved_landmarks[2][2]],
                ["RPA registered (voxel)", mri_registered_landmarks[2][1], mri_registered_landmarks[2][0], mri_registered_landmarks[2][2]],
                ["----------Landmarks found with non-normalized CT scan----------","nan","nan","nan"],
                ["Nasion improved (voxel)", ct_improved_landmarks[0][1], ct_improved_landmarks[0][0], ct_improved_landmarks[0][2]],
                ["Nasion registered (voxel)", ct_registered_landmarks[0][1], ct_registered_landmarks[0][0], ct_registered_landmarks[0][2]],
                ["LPA improved (voxel)", ct_improved_landmarks[1][1], ct_improved_landmarks[1][0], ct_improved_landmarks[1][2]],
                ["LPA registered (voxel)", ct_registered_landmarks[1][1], ct_registered_landmarks[1][0], ct_registered_landmarks[1][2]],
                ["RPA improved (voxel)", ct_improved_landmarks[2][1], ct_improved_landmarks[2][0], ct_improved_landmarks[2][2]],
                ["RPA registered (voxel)", ct_registered_landmarks[2][1], ct_registered_landmarks[2][0], ct_registered_landmarks[2][2]],
                ["If '-' appears in the line 'Dimensions not cropped', it means the original (uncropped) NIfTI file was used for the entire process.","nan","nan","nan"],
                ["If the cropped NIfTI file was used to retrieve the original coordinates of the LPA RPA and Nasion : subtract the z-dimension of the cropped file from that of the original file, and add the difference to the z-index. The x and y indices remain unchanged.","nan","nan","nan"]])
    
    if len(df_is_exist) != 0:
        if register_with_CT_not_normalized:
            data[1:7,:] = df_is_exist[1:7,:]
        else:
            data[8:14,:] = df_is_exist[8:14,:]
    
    if True: # Switches LPA and RPA if this axis was flipped during the identification process
        # mettre que ça switch le bon
        if register_with_CT_not_normalized:
            new_lpa = data[12:14,1:].copy()
            new_rpa = data[10:12,1:].copy()
            data[12:14,1:] = new_rpa
            data[10:12,1:] = new_lpa
        else:
            new_lpa = data[5:7,1:].copy()
            new_rpa = data[3:5,1:].copy()
            data[5:7,1:] = new_rpa
            data[3:5,1:] = new_lpa

    data = data.tolist()
    df_to_keep += data

    df = pd.DataFrame(df_to_keep)
    df.to_csv("test_2_registration.csv", index=False)

save_pts_to_csv()



        
















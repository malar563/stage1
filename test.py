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


print(len(arr))















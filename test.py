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
print(np.nonzero(binary)[0][0])




















    # def find_nose(self):
    #     for slice in range(0, len(self.skull[:,1,1])):
    #         pass

    #     iz, ix, iy = np.where(self.skull)
    #     x_center, y_center, z_center = int(np.mean(ix)), int(np.mean(iy)), int(np.mean(iz))
    #     print(x_center, y_center, z_center) # Le centre en z est inutile : pas à la hauteur du nez.


    # def binary_closing(self, iterations=2):

    #     # CETTE FONCTION NE MARCHE PAS. À JETER AUX POUBELLES

    #     from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, maximum_filter, generate_binary_structure, iterate_structure, label
    #     # from skimage.morphology import ball, binary_dilation, binary_erosion
    #     # from skimage.measure import label

    #     # self.skull = self.skull != 1
    #     # self.skull = np.where(self.skull, 1, 0)
    #     # self.skull = binary_closing(self.skull, iterations=iterations)
    #     # self.skull = self.skull != 1
    #     # self.skull = np.where(self.skull, 1, 0)
    #     # r2 = ball(2)
    #     # r3 = ball(3)
    #     iterations = 2
    #     struct = generate_binary_structure(3,1)

    #     erosion = binary_erosion(self.skull, structure=struct, iterations=iterations)
    #     erosion = np.where(erosion, 1, 0)

    #     labeled_array, num_of_structures = label(erosion, struct) # Associate a number to an island
    #     print(num_of_structures)
        
    #     ##jsp pk mais maximum filter ne marche pas
    #     # max_filter = maximum_filter(labeled_array, 5)
    #     # max_filter[labeled_array != 0] = labeled_array[labeled_array != 0]
    #     # self.skull = max_filter

    #     # #MOI AVANT
    #     dilation = binary_dilation(labeled_array, structure=struct, iterations=iterations+5)
    #     counts = np.bincount(dilation.ravel()) # Count the number of elements associated with each island (ascending number) 
    #     counts[0] = 0 # background count set to zero
    #     largest_label = np.argmax(counts) # Index of the maximum count = number given by np.label
    #     self.skull = labeled_array == largest_label
    #     dilation = np.where(dilation, 1, 0)
    #     self.skull = dilation * self.skull

    #     return self.skull
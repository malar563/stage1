import numpy as np
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
print(binary)
print(~binary)
binary = binary != 1
print(binary)

distance = distance_transform_edt(~binary)
print(distance)





















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
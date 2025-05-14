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

from scipy.ndimage import generate_binary_structure
from skimage.morphology import ball
balle = ball(2)
struct = generate_binary_structure(3, 1)
print(balle)
print(struct)


c = np.array([[1,2,3],[4,5,6],[7,8,9]])
c_flat = np.ravel(c)
print(c_flat)



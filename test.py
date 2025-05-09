import numpy as np
from scipy.ndimage import label


a = np.array([[0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [1,1,0,0,1,0],
              [0,0,0,1,1,0]])
labeled_array, num_features = label(a)
print(labeled_array, num_features)


x = np.bincount(np.array([1, 1, 1, 1, 1]))
print(x)

y = np.bincount(np.flip(np.array([0, 1, 1, 3, 2, 1, 7])))
print(y)


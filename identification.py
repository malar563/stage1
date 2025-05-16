import numpy as np
from segmentation import Segmentation
import matplotlib.pyplot as plt
# from segmentation import head # BEAUCOUP TROP LONG... SAVE EN PICKLE?


class Identification(Segmentation):

    def __init__(self):
        self.derive = None
        self.open_pickle()

    
    def fill_cavities(self):
        from scipy.ndimage import binary_fill_holes#, label, generate_binary_structure, binary_erosion, binary_dilation

        for i in range(0, len(self.head[1,1,:])):
            # Fill holes in the x-z plane 
            self.head[:,:,i] = binary_fill_holes(self.head[:,:,i])
            # self.head[:,i,:] = binary_fill_holes(self.head[:,i,:]) # Inutile : change peu de choses
            # Fill holes in the x-y plane
            try:
                self.head[i,:,:] = binary_fill_holes(self.head[i,:,:])
            except:
                continue
        return self.head


    def find_nasion(self, window_y=75):
        x_half_head = self.head[:,:int(len(self.head[1,:,1])//2),int(len(self.head[1,1,:])//2 - window_y):int(len(self.head[1,1,:])//2 + window_y)]
        index_z = np.arange(0, len(self.head[:,1,1]), 1)
        # print(index_z)
        counts_x = np.sum(x_half_head, axis = 1) # axis=2 donne somme en y, axis=0 donne somme en z
        plt.imshow(counts_x, origin="lower")
        plt.show()
        print(counts_x)
        self.head = x_half_head
        return self.head
    

    


id = Identification()
id.fill_cavities()
# id.show(id.head, 256, "y")
id.find_nasion()
id.show(id.head, 75, "y")
id.animation(id.head)


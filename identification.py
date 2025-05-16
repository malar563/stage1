import numpy as np
from segmentation import Segmentation
# from segmentation import head # BEAUCOUP TROP LONG... SAVE EN PICKLE?


class Identification(Segmentation):

    def __init__(self):
        self.derive = None
        self.open_pickle()

    
    def fill_cavities(self):
        from scipy.ndimage import binary_fill_holes#, label, generate_binary_structure, binary_erosion, binary_dilation

        # Fill holes in the x-y plane
        for i in range(0, len(self.head[1,1,:])):
            try:
                self.head[i,:,:] = binary_fill_holes(self.head[i,:,:])
            except:
                continue
            # Fill holes in the x-z plane 
            self.head[:,:,i] = binary_fill_holes(self.head[:,:,i])
            # self.head[:,i,:] = binary_fill_holes(self.head[:,i,:]) # Inutile : change peu de choses
        return self.head    

    


id = Identification()
# print(id.keep_face_surface())
id.fill_cavities()
id.show(id.head, 256, "y")

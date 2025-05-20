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


    def find_nasion(self):#75
        x_half_head = self.head[:,:int(len(self.head[1,:,1])//2),:]
        
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(x_half_head, axis = 1) # axis=2 donne somme en y, axis=0 donne somme en z

        # Index of the maximal values for each axis (new coordinate system)
        argmax_x = np.argmax(counts_x, axis=0) # En x
        # argmax_y = np.argmax(counts_x, axis=1) # En y

        # Finding x, y coordinate of the nose
        from scipy.ndimage import gaussian_filter1d
        argmax_x_filtered = gaussian_filter1d(argmax_x, sigma=3)
        deriv = np.gradient(argmax_x_filtered)
        
        nose_section = counts_x[:,np.argmin(deriv):np.argmax(deriv)]
        nose_y = int(np.mean(argmax_x[np.argmin(deriv):np.argmax(deriv)]))
        # Au lieu de faire une moyenne pour le nez en x, fiter une gaussienne? Mais est-ce que le nez a une forme gaussienne??
        nose_x = int((np.argmin(deriv)+np.argmax(deriv))//2) - np.argmin(deriv) 
        
        plt.imshow(nose_section, origin="lower")
        plt.scatter([nose_x], [nose_y], c="r")
        plt.show()
                
        print(nose_section[:,nose_x])
        diff = np.gradient(nose_section, axis=0) # axis=1 donne différentielle du sens transverse au nez

        diff = []
        for i in range(1, len(nose_section[:,nose_x])):
            diff.append(nose_section[i,nose_x]-nose_section[i-1,nose_x])
        # for i in range(1, len(nose_section[:,0])):
        #     diff.append(nose_section[i,0]-nose_section[i-1,0])
        print(diff)

        # plt.imshow(diff, origin="lower")
        # plt.show()
        # plt.plot(diff[:,nose_x])
        # plt.plot(gaussian_filter1d(diff, sigma=1))
        plt.plot(diff)
        plt.show()



        self.head = x_half_head
        return self.head
    

    


id = Identification()
id.fill_cavities()
# id.show(id.head, 256, "y")
id.find_nasion()
id.show(id.head, 75, "y")
id.animation(id.head)


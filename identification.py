

import numpy as np
from segmentation import Segmentation
import matplotlib.pyplot as plt
# from segmentation import head # BEAUCOUP TROP LONG... SAVE EN PICKLE?


class Identification(Segmentation):

    def __init__(self):
        # self.derivee = None
        self.open_pickle(file_name="head")
        self.nasion = None
        self.filled_y_slices = None

    
    def fill_cavities(self):
        from scipy.ndimage import binary_fill_holes#, label, generate_binary_structure, binary_erosion, binary_dilation

        self.filled_y_slices = []
        for i in range(0, len(self.head[1,1,:])):
            # Fill holes in the x-z and y-z plane 
            original = self.head[:,:,i]
            filled = binary_fill_holes(self.head[:,:,i])
            if not np.array_equal(original,filled):
                self.filled_y_slices.append(i)
            self.head[:,:,i] = filled
            self.head[:,i,:] = binary_fill_holes(self.head[:,i,:])
        print(self.filled_y_slices)
            # Fill holes in the x-y plane
        for i in range(0, len(self.head[:,1,1])):
            self.head[i,:,:] = binary_fill_holes(self.head[i,:,:])
        
        return self.head, self.filled_y_slices


    def find_nasion(self, y_min=150, y_max=350):
        x_half_head = self.head[:,:int(len(self.head[1,:,1])//2),y_min:y_max]
        
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(x_half_head, axis = 1) # axis=2 donne somme en y, axis=0 donne somme en z
        plt.imshow(counts_x, origin="lower")
        plt.show()

        # Index of the maximal values for y axis
        argmax_y = np.argmax(counts_x, axis=0)
        print(argmax_y)
        plt.plot(argmax_y)
        plt.show()

        # Finding y, z coordinate of the nose
        from scipy.ndimage import gaussian_filter1d
        argmax_y_filtered = gaussian_filter1d(argmax_y, sigma=3)#[100:400] # Cut to keep the middle zone
        # argmax_y = argmax_y[100:400]
        # nose_section = counts_x[:,100:400] 
        plt.plot(argmax_y_filtered)
        plt.show()
        derive = np.gradient(argmax_y_filtered)
        plt.plot(derive)
        plt.show()
        
        # Keep the middle part of the scan and find the z position of the tip of the nose
        nose_section = counts_x[:,np.argmin(derive):np.argmax(derive)] # AVANT CETAIT COUNTS_X ICI
        nose_z = int(np.mean(argmax_y[np.argmin(derive):np.argmax(derive)]))
        # print(nose_z)
        # # Au lieu de faire une moyenne pour le nez en y, fiter une gaussienne? Mais est-ce que le nez a une forme gaussienne??
        # nose_y = int((np.argmin(derive)+np.argmax(derive))//2) - np.argmin(derive)

        # Fit the x position of the nose tip
        from scipy.optimize import curve_fit
        def gaussian(x, height, position, std, offset):
            return height*np.exp(-((x-position)**2)/(2*std**2))+offset
        x = np.arange(0, len(nose_section[nose_z,:]))
        params, _ = curve_fit(gaussian, x, nose_section[nose_z,:])
        height, nose_y, std, offset = params
        nose_y = int(nose_y)
        
        plt.imshow(nose_section, origin="lower")
        plt.scatter([nose_y], [nose_z], c="b")
        plt.show()

        # Find the peak of the nasion
        from scipy.signal import find_peaks
        central_axis = nose_section[:,nose_y]
        nasion_z = find_peaks(-1*central_axis, prominence=5)[0][-1] # prominence = 4 : métal du nez disparaît, prominence = 6 : plus grande valeur possible
        plt.plot(-1*central_axis)
        plt.show()

        plt.imshow(nose_section, origin="lower")
        plt.scatter([nose_y], [nose_z], c="b")
        plt.scatter([nose_y], [nasion_z], c="r")
        plt.show()
    
        # Approximation : nose's y position = nasion's y position
        nasion_y = nose_y +  np.argmin(derive) + y_min # In self.head
        nasion_x = counts_x[nasion_z, nasion_y-y_min]
        nasion_x = np.nonzero(self.head[nasion_z,:,nasion_y])[0][0]
        
        # Nasion en (x, y, z)
        self.nasion = nasion_x, nasion_y, nasion_z
        return self.nasion
    

    def check_nasion(self):
        # # Si l'axe x passe à travers le nasion et règle de la main droite
        # # Plan axial : Valeur fixe de z
        plt.imshow(self.head[self.nasion[2],:,:], origin="lower")
        plt.scatter([self.nasion[1]], [self.nasion[0]], c="r")
        plt.show()
        # # Plan coronal : Valeur fixe de x
        plt.imshow(self.head[:,self.nasion[0],:], origin="lower")
        plt.scatter([self.nasion[1]], [self.nasion[2]], c="r")
        plt.show()
        # # Plan sagittal : Valeur fixe de y
        plt.imshow(self.head[:,:,self.nasion[1]], origin="lower")
        plt.scatter([self.nasion[0]], [self.nasion[2]], c="r")
        plt.show()

    
    def find_lpa_rpa(self):
        pass


    

id = Identification()
id.fill_cavities()
# id.show(id.head, 107, "y")
# id.show(id.head, 108, "y")
# id.show(id.head, 428, "y")
# id.show(id.head, 429, "y")
id.find_nasion()
id.check_nasion()
print(id.nasion)
id.show(id.head, 278, "y")
id.animation(id.head)



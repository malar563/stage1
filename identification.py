import numpy as np
from segmentation import Segmentation
import matplotlib.pyplot as plt
# from segmentation import head # BEAUCOUP TROP LONG... SAVE EN PICKLE?


class Identification(Segmentation):

    def __init__(self):
        # self.derivee = None
        self.open_pickle()
        self.nasion = None

    
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


    def find_nasion(self):
        x_half_head = self.head[:,:int(len(self.head[1,:,1])//2),:]
        
        # Sum up one values of the binary mask on the x axis (3D array -> 2D array)
        counts_x = np.sum(x_half_head, axis = 1) # axis=2 donne somme en y, axis=0 donne somme en z

        # Index of the maximal values for each axis (new coordinate system)
        argmax_x = np.argmax(counts_x, axis=0) # En x
        argmax_y = np.argmax(counts_x, axis=1) # En y
        # print(np.mean(argmax_y))

        # Finding x, y coordinate of the nose
        from scipy.ndimage import gaussian_filter1d
        argmax_x_filtered = gaussian_filter1d(argmax_x, sigma=3)
        derive = np.gradient(argmax_x_filtered)
        
        # Keep the middle part of the scan and find the y position of the tip of the nose
        nose_section = counts_x[:,np.argmin(derive):np.argmax(derive)]
        nose_y = int(np.mean(argmax_x[np.argmin(derive):np.argmax(derive)]))
        # print(nose_y)
        # # Au lieu de faire une moyenne pour le nez en x, fiter une gaussienne? Mais est-ce que le nez a une forme gaussienne??
        # nose_x = int((np.argmin(derive)+np.argmax(derive))//2) - np.argmin(derive)

        # Fit the x position of the nose tip
        from scipy.optimize import curve_fit
        def gaussian(x, height, position, std, offset):
            return height*np.exp(-((x-position)**2)/(2*std**2))+offset
        x = np.arange(0, len(counts_x[nose_y,:]))
        params, _ = curve_fit(gaussian, x, counts_x[nose_y,:])
        height, nose_x, std, offset = params
        # print(nose_x)
        nose_x = int(nose_x)
        
        plt.imshow(nose_section, origin="lower")
        plt.scatter([nose_x], [nose_y], c="b")
        plt.show()


        from scipy.signal import find_peaks
        central_axis = nose_section[:,nose_x]
        nasion_y = find_peaks(-1*central_axis, prominence=5)[0][-1] # prominence = 4 : métal du nez disparaît, prominence = 6 : plus grande valeur possible
        # print(nasion_y)
        plt.plot(-1*central_axis)
        plt.show()

        plt.imshow(nose_section, origin="lower")
        plt.scatter([nose_x], [nose_y], c="b")
        plt.scatter([nose_x], [nasion_y], c="r")
        plt.show()
    
        # ATTENTION!! Back to the previous coordinate system (z, x, y)
        # Approximation : nose's y position = nasion's y position 
        nasion_x = counts_x[nasion_y, nose_x + np.argmin(derive)]
        half_head_nasion_x = x_half_head[nasion_y,:,nasion_x]
        nasion_x = np.nonzero(half_head_nasion_x)[0][0]-1 # RAJOUT DU 1 POUR POGNER L'EXTÉRIEUR?
        print(nasion_x)
        
        nasion_z = nasion_y
        nasion_y = nose_x +  np.argmin(derive)
        
        # Nasion en (x, y, z)
        self.nasion = nasion_x, nasion_y, nasion_z # (x, y, z)
        # NASION_X EST ISHH BON : ÇA DEVRAIT DONNER ENVIRON 51 MAIS ÇA DONNE 56
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
        # # Plan coronal : Valeur fixe de y
        plt.imshow(self.head[:,:,self.nasion[1]], origin="lower")
        plt.scatter([self.nasion[0]], [self.nasion[2]], c="r")
        plt.show()


    

id = Identification()
id.fill_cavities()
# id.show(id.head, 256, "y")
id.find_nasion()
id.check_nasion()
print(id.nasion)
id.show(id.head, 278, "y")
id.animation(id.head)


import matplotlib.pyplot as plt
import numpy as np


class Segmentation:
    
    def __init__(self):
        self.array = None
        self.resolution = None
        self.px_spacing = None
        self.load_file()
        self.masked_array = None
        self.head = None
        self.skull = None
        # Trouver un moyen de mettre le nom de fichier automatique


    # Trouver un moyen de mettre le range automatique selon ce que le fichier contient (mettre des try/except)
    def load_file(self):
        import pydicom as dicom
        dcms = []
        for i in range(4, 605):
            path = r"DICOM_010\COW_Angio_0.6_Hv36_3" + f"\I{i}.dcm"
            dcm_file = dicom.dcmread(path)

            # dcms.append(dcm_file.pixel_array)

            image = dcm_file.pixel_array.astype(np.int16)
            intercept = float(dcm_file.RescaleIntercept)
            slope = float(dcm_file.RescaleSlope)
            dcms.append(image * slope + intercept)
        
            
        # (0018,0050) Slice Thickness                     DS: '0.6'
        print(dcm_file)
        resolution = dcm_file[0x0018, 0x0050].value
        px_spacing = dcm_file[0x0028, 0x0030].value

        self.array = np.array(dcms)
        self.resolution = resolution
        self.px_spacing = px_spacing
        return self.array, self.resolution, self.px_spacing


    def cut(self):
        # Cut from the top of the CT scan depending on the resolution
        if self.resolution == 0.6:
            self.array = self.array[-256:,:,:]
        else :
            self.array = self.array[-512:,:,:]
        return self.array


    def show(self, array, slice, axis):
        # # Si l'axe x passe à travers le nasion et règle de la main droite
        if axis == "z":
            # # Plan axial : Valeur fixe de z
            plt.imshow(array[slice,:,:], cmap='gist_gray', origin="lower")
        elif axis == "x":
            # # Plan coronal : Valeur fixe de x
            plt.imshow(array[:,slice,:], cmap='gist_gray', origin="lower")
        elif axis == "y":
            # # Plan coronal : Valeur fixe de y
            plt.imshow(array[:,:,slice], cmap='gist_gray', origin="lower")
        else:
            raise TypeError("Must be x, y or z axis")
        plt.show()


    def apply_threshold(self, threshold_head=-200, threshold_skull=300):
        # Array with "True" where it is, and "False" where it is not
        thresholded_head = self.array >= threshold_head
        thresholded_skull = self.array >= threshold_skull
        # # Put the value of the threshold if True, and -1000 if False
        # mask = np.where(thresholded_scan, threshold, -1000)
        # Put the value 1 if True, and 0 if False
        self.head = np.where(thresholded_head, 1, 0)
        self.skull = np.where(thresholded_skull, 1, 0)
        return self.head, self.skull
    
    
# Mettre ça mieux pour que ça se répète pas
    def keep_largest_island(self):
        from scipy.ndimage import label, generate_binary_structure

        s = np.where(generate_binary_structure(3,3), 1, 0) # Define the connection between elements

        labeled_array_head, num_of_structures_head = label(self.head, s) # Associate a number to an island
        counts = np.bincount(labeled_array_head.ravel()) # Count the number of elements associated with each island (ascending number) 
        counts[0] = 0 # background count set to zero
        largest_label_head = np.argmax(counts) # Index of the maximum count = number given by np.label
        self.head = labeled_array_head == largest_label_head
        
        labeled_array_skull, num_of_structures_head = label(self.skull, s) # Associate a number to an island
        # plt.imshow(labeled_array_skull[:,:,230], cmap="viridis", origin="lower")
        # plt.show()
        counts = np.bincount(labeled_array_skull.ravel()) # Count the number of elements associated with each island (ascending number) 
        counts[0] = 0 # background count set to zero
        largest_label_skull = np.argmax(counts) # Index of the maximum count = number given by np.label
        self.skull = labeled_array_skull == largest_label_skull

        return self.head, self.skull
    

    def fill_holes(self):
        from scipy.ndimage import binary_fill_holes

        self.skull = binary_fill_holes(self.skull)
        return self.skull


    def find_nose(self):
        for slice in range(0, len(self.skull[:,1,1])):
            pass


        iz, ix, iy = np.where(self.skull)
        x_center, y_center, z_center = int(np.mean(ix)), int(np.mean(iy)), int(np.mean(iz))
        print(x_center, y_center, z_center) # Le centre en z est inutile : pas à la hauteur du nez.


    def save_to_dicom(self):
        import pydicom as dicom
        dcms = []
        for i in range(4, 605):
            path = r"DICOM_010\COW_Angio_0.6_Hv36_3" + f"\I{i}.dcm"
            dcm_file = dicom.dcmread(path)

            # dcms.append(dcm_file.pixel_array)

            image = dcm_file.pixel_array.astype(np.int16)
            intercept = float(dcm_file.RescaleIntercept)
            slope = float(dcm_file.RescaleSlope)
            dcms.append(image * slope + intercept)
            dcm_file.save_as('out.dcm')
        

        self.array = np.array(dcms)
        
        pass


    def wrap_solidify(self, max_gap_mm=30):
        from scipy.ndimage import binary_dilation, binary_erosion
        allo = binary_dilation(self.skull, iterations=10)
        allo2 = binary_erosion(allo, iterations=10)
        self.skull = allo2

        return self.skull







    
# Segmentation
ct_scan = Segmentation()
print("Resolution", ct_scan.resolution, ct_scan.px_spacing)
ct_scan.cut()
print("Volume shape", ct_scan.array.shape)

ct_scan.apply_threshold()
ct_scan.keep_largest_island()

# ya une ligne qui touche où le nez pour ct_scan.head qui ne s'en va pas (sur 3D slicer non plus)
ct_scan.show(ct_scan.skull, 256, "y")
ct_scan.fill_holes()
ct_scan.show(ct_scan.skull, 256, "y")
ct_scan.wrap_solidify()
ct_scan.show(ct_scan.skull, 256, "y")
# ct_scan.show(ct_scan.skull, 268, "y")

# ct_scan.show(ct_scan.head, 232, "x")
ct_scan.show(ct_scan.head, 270, "y")
ct_scan.show(ct_scan.head, 131, "z")

# Il faut que je fasse une fonction pour trouver un moyen de séparer le bout de métal qui touche au nez
ct_scan.find_nose()


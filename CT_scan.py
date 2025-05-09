import matplotlib.pyplot as plt
import numpy as np
# import scikit
# import nibabel as nib


class Segmentation:
    
    def __init__(self):
        self.array = None
        self.resolution = None
        self.load_file()
        self.masked_array = None
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
        
        # print(dcm_file)
        # (0018,0050) Slice Thickness                     DS: '0.6'
        resolution = dcm_file[0x0018, 0x0050].value

        self.array = np.array(dcms)
        self.resolution = resolution
        return self.array, self.resolution


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


    def apply_threshold(self, threshold=-200):
        # Array with "True" where it is, and "False" where it is not
        thresholded_scan = self.array >= threshold
        # # Put the value of the threshold if True, and -1000 if False
        # mask = np.where(thresholded_scan, threshold, -1000)
        # Put the value 1 if True, and 0 if False
        self.masked_array = np.where(thresholded_scan, 1, 0)
        return self.masked_array
    

    def keep_largest_island(self, ct_scan):
        from scipy.ndimage import label, generate_binary_structure
    
        s = np.where(generate_binary_structure(3,3), 1, 0)

        labeled_array, num_of_structures = label(ct_scan, s)
        print(labeled_array, num_of_structures)
        return labeled_array, num_of_structures, s
    


ct_scan = Segmentation()
print(ct_scan.resolution)
ct_scan.cut()
print("Volume shape", ct_scan.array.shape)

ct_scan.show(ct_scan.array, 100, "z")

ct_scan.apply_threshold(-200)
ct_scan.show(ct_scan.masked_array, 100, "x")

ct_scan.keep_largest_island(ct_scan.masked_array)


import matplotlib.pyplot as plt
import numpy as np
import os


class Segmentation:
    
    def __init__(self, folder_path="DICOM_010/COW_Angio_0.6_Hv36_3"):
        self.array = None
        self.resolution = None
        self.px_spacing = None
        self.folder_path = folder_path
        self.load_file()
        self.masked_array = None
        self.head = None
        self.skull = None
        self.air = None    


    def load_file(self, max_files=2000):
        import pydicom as dicom

        dcms = []
        for i in range(1, max_files):
            filename = f"I{i}.dcm"
            path = os.path.join(self.folder_path, filename)
            try:
                dcm_file = dicom.dcmread(path)
                # dcms.append(dcm_file.pixel_array)
                image = dcm_file.pixel_array.astype(np.int16)
                intercept = float(dcm_file.RescaleIntercept)
                slope = float(dcm_file.RescaleSlope)
                dcms.append(image * slope + intercept)
            except:
                continue
        # print(dcm_file.RescaleIntercept, dcm_file.RescaleSlope)
        resolution = dcm_file[0x0018, 0x0050].value
        px_spacing = dcm_file[0x0028, 0x0030].value

        self.array = np.array(dcms)
        self.resolution = resolution
        self.px_spacing = px_spacing
        return self.array, self.resolution, self.px_spacing
    

        # À finir
    def save_to_dicom(self):
        import pydicom as dicom
        dcms = []
        for i in range(0, len(self.skull[:,1,1])-1):
            filename = f"I100.dcm" # Doesn't matter which one
            path = os.path.join(self.folder_path, filename)
            path = self.folder_path
            dcm_file_head = dicom.dcmread(path)
            dcm_file_skull = dicom.dcmread(path)

            # Je dois mettre la slice en z pour avoir une array 2D
            dcm_file_head.PixelData = self.skull[i,:,:].tobytes()
            dcm_file_skull.PixelData = self.head[i,:,:].tobytes()

            # Est-ce que ya juste l'array qui change d'un fichier dicom à l'autre?
            dcm_file_head.save_as(f"head{i}.dcm")
            dcm_file_skull.save_as(f'skull{i}.dcm')

        
    def show(self, array, slice, axis):
        # # Si l'axe x passe à travers le nasion et règle de la main droite
        if axis == "z":
            # # Plan axial : Valeur fixe de z
            plt.imshow(array[slice,:,:], cmap='gist_gray', origin="lower")
        elif axis == "x":
            # # Plan coronal : Valeur fixe de x
            plt.imshow(array[:,slice,:], cmap='gist_gray', origin="lower")
        elif axis == "y":
            # # Plan sagittal : Valeur fixe de y
            plt.imshow(array[:,:,slice], cmap='gist_gray', origin="lower")
        else:
            raise TypeError("Must be x, y or z")
        plt.show()


    # Marche mal
    def animation(self, img):
        import plotly.express as px
        fig = px.imshow(img, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
        fig.show()


    def save_to_pickle(self, object=None, file_name = "head"):
        import pickle
        if object is None:
            object = self.head
        with open(file_name+".pickle", "wb") as f:  # "wb" = write binary
            pickle.dump(object, f)


    def open_pickle(self, file_name="head"):
        import pickle
        with open(file_name+".pickle", "rb") as f:
            self.head = pickle.load(f) # Faudrait que self.head ait un nom modifiable
            self.head = np.where(self.head, 1, 0)


    def cut(self):
        # Cut from the top of the CT scan depending on the resolution
        if self.resolution == 0.6:
            self.array = self.array[-256:,:,:]
        else :
            self.array = self.array[-512:,:,:]
        return self.array


    def apply_threshold(self, threshold_head=-200, threshold_skull=200, threshold_no_arteries = 500):
        # Array with "True" where it is, and "False" where it is not
        thresholded_head = self.array >= threshold_head
        thresholded_air = self.array <= threshold_head
        thresholded_skull = self.array >= threshold_skull
        thresholded = self.array >= threshold_no_arteries
        # Put the value 1 if True, and 0 if False
        self.head = np.where(thresholded_head, 1, 0)
        self.air = np.where(thresholded_air, 1, 0)
        self.skull = np.where(thresholded_skull, 1, 0)
        self.masked_array = np.where(thresholded, 1, 0)
        return self.head, self.skull, self.masked_array, self.air
    
    
    def keep_largest_island(self):
        from scipy.ndimage import label, generate_binary_structure

        def largest_connected_island(mask):
            s = generate_binary_structure(3, 3)
            labeled, _ = label(mask, s) # Associate a number to an island
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # ignore background
            return labeled == np.argmax(counts) # Index of the maximum count = number given by np.label

        self.head = largest_connected_island(self.head)
        self.skull = largest_connected_island(self.skull)
        self.masked_array = largest_connected_island(self.masked_array)

        return self.head, self.skull, self.masked_array
    

    def fill_holes(self):
        from scipy.ndimage import binary_fill_holes

        self.skull = binary_fill_holes(self.skull)
        return self.skull


    def remove_arteries(self, max_distance = 3): # Mettre 200 et 500 comme seuil avec cette distance
        from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure

        self.masked_array = self.masked_array != 1
        distance = distance_transform_edt(self.masked_array)
        close_to_bone = distance < max_distance
        self.skull = self.skull & close_to_bone
        self.skull = binary_dilation(self.skull, generate_binary_structure(3, 1))

        return self.skull
    



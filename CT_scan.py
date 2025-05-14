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
        self.air = None
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
        # print(dcm_file.RescaleIntercept, dcm_file.RescaleSlope)
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


# Mettre ça mieux pour que ça se répète pas
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
            raise TypeError("Must be x, y or z")
        plt.show()


# Mettre ça mieux pour que ça se répète pas
    def apply_threshold(self, threshold_head=-200, threshold_skull=200):
        # Array with "True" where it is, and "False" where it is not
        thresholded_head = self.array >= threshold_head
        thresholded_air = self.array <= threshold_head
        thresholded_skull = self.array >= threshold_skull
        thresholded = np.logical_and(self.array >= threshold_head, self.array <= threshold_skull) #self.array <= threshold_skull
        # Put the value 1 if True, and 0 if False
        self.head = np.where(thresholded_head, 1, 0)
        self.air = np.where(thresholded_air, 1, 0)
        self.skull = np.where(thresholded_skull, 1, 0)
        self.masked_array = np.where(thresholded, 1, 0)
        return self.head, self.skull, self.masked_array, self.air
    
    
# Mettre ça mieux pour que ça se répète pas
    def keep_largest_island(self):
        from scipy.ndimage import label, generate_binary_structure

        s = np.where(generate_binary_structure(3,3), 1, 0) # Define the connection between elements

        labeled_array_head, num_of_structures_head = label(self.head, s) # Associate a number to an island
        counts = np.bincount(labeled_array_head.ravel()) # Count the number of elements associated with each island (ascending number) 
        counts[0] = 0 # background count set to zero
        largest_label_head = np.argmax(counts) # Index of the maximum count = number given by np.label
        self.head = labeled_array_head == largest_label_head
        
        labeled_array_skull, num_of_structures_skull = label(self.skull, s) # Associate a number to an island
        # plt.imshow(labeled_array_skull[:,:,230], cmap="viridis", origin="lower")
        # plt.show()
        counts = np.bincount(labeled_array_skull.ravel()) # Count the number of elements associated with each island (ascending number) 
        counts[0] = 0 # background count set to zero
        largest_label_skull = np.argmax(counts) # Index of the maximum count = number given by np.label
        self.skull = labeled_array_skull == largest_label_skull

        labeled_array, num_of_structures = label(self.masked_array, s) # Associate a number to an island
        counts = np.bincount(labeled_array.ravel()) # Count the number of elements associated with each island (ascending number) 
        counts[0] = 0 # background count set to zero
        largest_label = np.argmax(counts) # Index of the maximum count = number given by np.label
        self.masked_array = labeled_array == largest_label

        return self.head, self.skull, self.masked_array
    

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


    # À finir
    def save_to_dicom(self):
        import pydicom as dicom
        dcms = []
        # Mettre chemin adaptable + un range qui se modifie automatique
        for i in range(0, len(self.skull[:,1,1])-1):
            path = r"DICOM_010\COW_Angio_0.6_Hv36_3\I15.dcm"
            dcm_file_head = dicom.dcmread(path)
            dcm_file_skull = dicom.dcmread(path)

            # Je dois mettre la slice en z pour avoir une array 2D
            dcm_file_head.PixelData = self.skull[i,:,:].tobytes()
            dcm_file_skull.PixelData = self.head[i,:,:].tobytes()

            # Est-ce que ya juste l'array qui change d'un fichier dicom à l'autre?
            dcm_file_head.save_as(f"head{i}.dcm")
            dcm_file_skull.save_as(f'skull{i}.dcm')
        

    def binary_closing(self, iterations=2):
        from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, generate_binary_structure, iterate_structure#, label
        from skimage.morphology import ball, erosion, dilation
        from skimage.measure import label

        # self.skull = self.skull != 1
        # self.skull = np.where(self.skull, 1, 0)
        # self.skull = binary_closing(self.skull, iterations=iterations)
        # self.skull = self.skull != 1
        # self.skull = np.where(self.skull, 1, 0)


        r2 = ball(2)
        r3 = ball(3)

        # erosion = binary_erosion(self.skull, structure=r2)
        # erosion = np.where(erosion, 1, 0)
        # print(erosion.shape)

        # labeled_array, num_of_structures = label(erosion, r2) # Associate a number to an island
        
        # dilation = binary_dilation(labeled_array, structure=r3)
        labeled_skull = erosion(self.skull, r2)[0]
        labeled_skull = np.where(labeled_skull, 1, 0)
        print(labeled_skull)

        labeled_array, num_of_structures = label(labeled_skull, r2) # Associate a number to an island
        
        
        labeled_skull = dilation(labeled_array, r3)


        self.skull = labeled_skull
        self.skull = np.where(self.skull, 1, 0)

        # radius = 5
        # volume = self.skull
        # struct = generate_binary_structure(3, 1)  # 3D connectivity
        # struct = iterate_structure(struct, radius)
        # # Apply morphological closing
        # closed = binary_closing(volume, structure=struct)
        # self.skull = closed


        # min_size = 500
        # volume = self.skull
        # labeled, num = label(volume)
        # output = np.zeros_like(volume)
        # for i in range(1, num + 1):
        #     component = (labeled == i)
        #     if component.sum() >= min_size:
        #         output[component] = 1
        # self.skull = output

        return self.skull
    

    def test(self):
        from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
        iter = 13
        self.masked_array = binary_erosion(self.masked_array, iterations=iter)
        self.masked_array = binary_dilation(self.masked_array, iterations=iter+2)
        self.masked_array = binary_erosion(self.masked_array, iterations=3)


        return self.masked_array


    def animation(self):
        import plotly.express as px
        img = ct_scan.skull
        fig = px.imshow(img, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
        fig.show()
    
    






    
# Segmentation
ct_scan = Segmentation()
print("Resolution", ct_scan.resolution, ct_scan.px_spacing)
ct_scan.cut()
print("Volume shape", ct_scan.array.shape)


ct_scan.apply_threshold()
ct_scan.keep_largest_island()

ct_scan.show(ct_scan.air, 256, "y")

# ct_scan.test()
# ct_scan.keep_largest_island()
# ct_scan.show(ct_scan.masked_array, 256, "x")



# ya une ligne qui touche où le nez pour ct_scan.head qui ne s'en va pas (sur 3D slicer non plus)
ct_scan.show(ct_scan.skull, 156, "z")
ct_scan.fill_holes()
# ct_scan.animation()
ct_scan.show(ct_scan.skull, 256, "y")
ct_scan.binary_closing()
# ct_scan.keep_largest_island()
ct_scan.show(ct_scan.skull, 156, "z")
ct_scan.animation()





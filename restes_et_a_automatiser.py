import numpy as np

# x, y, z
# LPA : 6_cow_hv36
resolution = [0.486328125, 0.486328125, 0.6000000238418579]
pt_registre = (223, 82, 63)
pt_estime = (237, 82, 57)
# RPA : 6_cow_hv36
pt_registre = (228, 404, 69)
pt_estime = (237, 404, 70)

# LPA : 301_carotid_0625mm
resolution = [0.4882810115814209, 0.4882810115814209, 0.3125]
# Parfaitement placé
# RPA : 301_carotid_0625mm
pt_registre = (251, 401, 135)
pt_estime = (265, 399, 130)



def compute_3D_distance(res=resolution, pt_reg=pt_registre, pt_est=pt_estime):
    return np.sqrt((res[0]*(pt_est[0]-pt_reg[0]))**2 + (res[1]*(pt_est[1]-pt_reg[1]))**2 + (res[2]*(pt_est[2]-pt_reg[2]))**2)

print(compute_3D_distance())





















    # def find_nose(self):
    #     for slice in range(0, len(self.skull[:,1,1])):
    #         pass

    #     iz, ix, iy = np.where(self.skull)
    #     x_center, y_center, z_center = int(np.mean(ix)), int(np.mean(iy)), int(np.mean(iz))
    #     print(x_center, y_center, z_center) # Le centre en z est inutile : pas à la hauteur du nez.


    # def binary_closing(self, iterations=2):

    #     # CETTE FONCTION NE MARCHE PAS. À JETER AUX POUBELLES

    #     from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, maximum_filter, generate_binary_structure, iterate_structure, label
    #     # from skimage.morphology import ball, binary_dilation, binary_erosion
    #     # from skimage.measure import label

    #     # self.skull = self.skull != 1
    #     # self.skull = np.where(self.skull, 1, 0)
    #     # self.skull = binary_closing(self.skull, iterations=iterations)
    #     # self.skull = self.skull != 1
    #     # self.skull = np.where(self.skull, 1, 0)
    #     # r2 = ball(2)
    #     # r3 = ball(3)
    #     iterations = 2
    #     struct = generate_binary_structure(3,1)

    #     erosion = binary_erosion(self.skull, structure=struct, iterations=iterations)
    #     erosion = np.where(erosion, 1, 0)

    #     labeled_array, num_of_structures = label(erosion, struct) # Associate a number to an island
    #     print(num_of_structures)
        
    #     ##jsp pk mais maximum filter ne marche pas
    #     # max_filter = maximum_filter(labeled_array, 5)
    #     # max_filter[labeled_array != 0] = labeled_array[labeled_array != 0]
    #     # self.skull = max_filter

    #     # #MOI AVANT
    #     dilation = binary_dilation(labeled_array, structure=struct, iterations=iterations+5)
    #     counts = np.bincount(dilation.ravel()) # Count the number of elements associated with each island (ascending number) 
    #     counts[0] = 0 # background count set to zero
    #     largest_label = np.argmax(counts) # Index of the maximum count = number given by np.label
    #     self.skull = labeled_array == largest_label
    #     dilation = np.where(dilation, 1, 0)
    #     self.skull = dilation * self.skull

    #     return self.skull
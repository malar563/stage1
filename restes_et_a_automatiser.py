import numpy as np

# x, y, z
# LPA : 6_cow_hv36
resolution = [0.486328125, 0.486328125, 0.6000000238418579]
pt_registre = (223, 86, 62)
pt_estime = (237, 82, 57) # <-----
# RPA : 6_cow_hv36
pt_registre = (228, 400, 70)
pt_estime = (237, 404, 70) # <-----

# # LPA : 301_carotid_0625mm
# resolution = [0.4882810115814209, 0.4882810115814209, 0.3125]
# # Parfaitement placé
# # RPA : 301_carotid_0625mm
# pt_registre = (251, 401, 135)
# pt_estime = (265, 399, 130)




# 01-07-2025 (z,x,y)

# 301_carotid_angion_0625mm
# rpa_reg : (122, 252, 397) -> rpa_amélioré : (122, 252, 399) -------- PARFAIT avec amélioration
# lpa_reg : (122, 233, 88) -> lpa_amélioré : (122, 233, 59) -------- PARFAIT avec registré (oreille pliée)

# 6_cow_angio__06__hv36__3
# rpa_reg : (70, 228, 399) -> rpa_amélioré : (70, 228, 400) -------- MIEUX avec amélioration (voir erreur en haut)
# lpa_reg : (62, 223, 84) -> lpa_amélioré : (62, 223, 86) -------- MIEUX avec registré

# 605_sag_1mm CONTINUER ICIIII
# rpa_reg : (65, 280, 206) -> rpa_amélioré : (65, 280, 207) -------- PARFAIT avec amélioration
# lpa_reg : (72, 267, 51) -> lpa_amélioré : (72, 267, 53) -------- PARFAIT avec registré

# 6_cta_thins
# rpa_reg : (52, 267, 414) -> rpa_amélioré : (52, 267, 453) -------- PARFAIT avec registration (Métal)
# lpa_reg : (44, 251, 95) -> lpa_amélioré : (44, 251, 98) -------- PARFAIT avec amélioration

# 311_bone-mar
# rpa_reg : (37, 266, 404) -> rpa_amélioré : (37, 266, 406) -------- (38,258,406) <----------------------------------------------------
# lpa_reg : (34,277,102) -> lpa_amélioré : (34,277,102) -------- (23,294,105) <----------------------------------------------------
# x, y, z
resolution = [0.5, 0.5, 0.625]
# RPA
pt_registre = (266, 406,37)
pt_estime = (258,406,38) # <-----
# # LPA :
pt_registre = (277,102,34)
pt_estime = (294,105,23) # <-----


# 3_ax_head_std_thin
# rpa_reg : (26,262,425) -> rpa_amélioré : (26,262,426) -------- (30,266,426)
# lpa_reg : (23,269,89) -> lpa_amélioré : (23,269,92) -------- (26,274,91)
# x, y, z
resolution = [0.429688,0.429688,1.25]
# RPA
pt_registre = (262,426,26)
pt_estime = (266,426,30) # <-----
# # LPA :
pt_registre = (269,92,23)
pt_estime = (274,91,26) # <-----

# 306_70_kev_bone_plus
# rpa_reg : (39,267,404) -> rpa_amélioré : (39,267,406) -------- PAREIL QUE 311_bone-mar
# lpa_reg : (36,271,102) -> lpa_amélioré : (36,271,105) -------- PAREIL QUE 311_bone-mar

# 301_70_kev_stnd
# rpa_reg : (39,267,404) -> rpa_amélioré : (39,267,406) -------- PAREIL QUE 311_bone-mar
# lpa_reg : (36,271,102) -> lpa_amélioré : (36,271,105) -------- PAREIL QUE 311_bone-mar

# 6_thin_with_dmpr
# rpa_reg : (49,262,409) -> rpa_amélioré : (44, 259, 410) -------- (55, 266, 411)
# lpa_reg : (42,270,74) -> lpa_amélioré : (42,270,77) -------- (46,274,75)
# x, y, z
resolution = [0.429688,0.429688,0.625]
# RPA
pt_registre = (262,409,44)
pt_estime = (266, 411,55) # <-----
# # LPA :
pt_registre = (270,77,42)
pt_estime = (274,75,46) # <-----

# 311_bone-mar
# rpa_reg : (44, 259, 407) -> rpa_amélioré : (49,262,410) -------- PARFAIT avec registration
# lpa_reg : (27, 288, 101) -> lpa_amélioré : (27, 288, 107) -------- PARFAIT avec amélioration


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







            # self.lpa = self.registered_lpa

        # # Ya absolument rien qui marche ci-bas donc inutile
        # lpa_y_final = self.__find_lpa_y__(lpa_y=lpa_y)
        # print(lpa_y_final)
        # ROI_lpa = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y_final-window:lpa_y_final+window]
        # # frame_before_filling = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y_final]
        # # frame_start_filling = self.head[lpa_z-window:lpa_z+window,lpa_x-window:lpa_x+window,lpa_y_final+1]
        # frame_before_filling = self.head[:,:,lpa_y_final]
        # frame_start_filling = self.head[:,:,lpa_y_final+1]
        # intersection = ~frame_before_filling & frame_start_filling
        # plt.imshow(intersection, origin="lower")
        # plt.scatter([lpa_x], [lpa_z], c="r")
        # plt.show()

        
        # self.show_3D_array(ROI_lpa, axis=2)
        # counts_x = np.sum(ROI_lpa, axis=2) # axis=2 donne somme en y, axis=0 donne somme en z
        # plt.imshow(counts_x, origin="lower")
        # plt.show()


        
    # def __find_lpa_y__(self, lpa_y):
    #     self.filled_y_slices = np.array(self.filled_y_slices)

    #     if lpa_y in self.filled_y_slices:
    #         print(self.filled_y_slices)   
    #         index_lpa_y = np.where(self.filled_y_slices == lpa_y)[0][0]
    #         print(index_lpa_y)
    #         for i in range(1, index_lpa_y):
    #             if self.filled_y_slices[index_lpa_y-i] != lpa_y-i:
    #                 return self.filled_y_slices[index_lpa_y-i+1]-1 # -1 to get the surface of the head
    #         return self.filled_y_slices[0] - 1 # -1 to get the surface of the head
    #     else:
    #         index_lpa = np.argmin(np.abs(self.filled_y_slices-lpa_y))
    #         return self.filled_y_slices[index_lpa]-1 # -1 to get the surface of the head
        
    # def __find_rpa_y__(self, rpa_y):
    #     self.filled_y_slices = np.array(self.filled_y_slices)

    #     if rpa_y in self.filled_y_slices:
    #         print(self.filled_y_slices)   
    #         index_rpa_y = np.where(self.filled_y_slices == rpa_y)[0][0]
    #         print(index_rpa_y)
    #         for i in range(1, len(self.filled_y_slices)-index_rpa_y):
    #             if self.filled_y_slices[index_rpa_y+i] != rpa_y+i:
    #                 return self.filled_y_slices[index_rpa_y+i-1]+1 # -1 to get the surface of the head
    #         return self.filled_y_slices[-1] + 1 # -1 to get the surface of the head
    #     else:
    #         index_lpa = np.argmin(np.abs(self.filled_y_slices-rpa_y))
    #         return self.filled_y_slices[index_lpa]+1 # -1 to get the surface of the head
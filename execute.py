from segmentation import Segmentation


# Segmentation
ct_scan = Segmentation()
# ct_scan = Segmentation(folder_path="DICOM_003/Carotid_Angio_0.625mm")
print("Resolution", ct_scan.resolution, ct_scan.px_spacing)
ct_scan.cut()
print("Volume shape", ct_scan.array.shape)

ct_scan.apply_threshold()
ct_scan.keep_largest_island()

# ya une ligne qui touche où le nez pour ct_scan.head qui ne s'en va pas (sur 3D slicer non plus)
# ct_scan.show(ct_scan.skull, 256, "y")
ct_scan.fill_holes()
# ct_scan.show(ct_scan.skull, 256, "y")
# ct_scan.animation(ct_scan.skull)
ct_scan.remove_arteries()
ct_scan.fill_holes()
# ct_scan.show(ct_scan.skull, 256, "y")
# ct_scan.animation(ct_scan.skull)


# Trouver un moyen de ne pas toujours avoir besoin de re-rouler "segmentation" pour avoir "head"
head = ct_scan.head
ct_scan.save_to_pickle(file_name="head")

# ct_scan.show(head, 256, "y")



# Si avec ants :
# 1 - Convertir tous les fichiers dicom en .nii avec nifti.py
# 2 - 
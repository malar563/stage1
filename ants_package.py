import ants
import matplotlib.pyplot as plt


img1 = ants.image_read('nifti/6_cow_angio__06__hv36__3.nii.gz')
img2 = ants.image_read('nifti/301_carotid_angio_0625mm.nii.gz').numpy()

print(img2)

plt.imshow(img2[250,:,:], origin="lower")
plt.show()
plt.imshow(img2[:,250,:], origin="lower")
plt.show()
plt.imshow(img2[:,:,700], origin="lower")
plt.show()
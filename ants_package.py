import ants
import matplotlib.pyplot as plt

# TUTORIEL : https://github.com/ANTsX/ANTsPy/blob/master/tutorials/10minTutorial.ipynb
# You can read also convert numpy arrays to ANTsImage types.. Here's an example of an fMRI image (an image with "components")

# arr_4d = np.random.randn(70,70,70,10).astype('float32')
# img_fmri = ants.from_numpy(arr_4d, has_components=True)
# print(img_fmri)

# Segmentation (je vois pas en quoi c'est mieux que ce que j'ai fait... À explorer)
img = ants.image_read(ants.get_ants_data('r16'))
img = ants.resample_image(img, (64,64), 1, 0)
mask = ants.get_mask(img)
img_seg = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', 
                       i='kmeans[3]', x=mask)
print(img_seg.keys())
ants.plot(img_seg['segmentation'])


img = ants.image_read( ants.get_ants_data('r16') ,2)
mask = ants.get_mask( img ).threshold_image( 1, 2 )
segs=ants.atropos( a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
thickimg = ants.kelly_kapowski(s=segs['segmentation'], g=segs['probabilityimages'][1],
                            w=segs['probabilityimages'][2], its=45, 
                            r=0.5, m=1)
print(thickimg)
img.plot(overlay=thickimg, overlay_cmap='jet')

# Registration (ce que moi j'ai besoin : réorienter)
fixed = ants.image_read( ants.get_ants_data('r16') ).resample_image((64,64),1,0)
moving = ants.image_read( ants.get_ants_data('r64') ).resample_image((64,64),1,0)
fixed.plot(overlay=moving, title='Before Registration')
mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN' )
print(mytx)
warped_moving = mytx['warpedmovout']
fixed.plot(overlay=warped_moving,
           title='After Registration')


# Move to & from nibabel images with ants.to_nibabel() and ants.from_nibabel() # Utile pour moi?





# TUTORIEL FINI
img = ants.image_read('nifti/6_cow_angio__06__hv36__3.nii.gz')
img2 = ants.image_read('nifti/301_carotid_angio_0625mm.nii.gz').numpy()


ants.plot(img, overlay = img > img.mean())

# # Weird, marche pas mm si c'est comme dans le tutoriel
# img = ants.smooth_image(img, 2)
# plt.imshow(img[250,:,:], origin="lower")
# plt.show()
# plt.imshow(img[:,250,:], origin="lower")
# plt.show()
# plt.imshow(img[:,:,700], origin="lower")
# plt.show()
# img = ants.resample_image(img, (3,3,3))
# plt.imshow(img[250,:,:], origin="lower")
# plt.show()
# plt.imshow(img[:,250,:], origin="lower")
# plt.show()
# plt.imshow(img[:,:,700], origin="lower")
# plt.show()

print(img2)

plt.imshow(img2[250,:,:], origin="lower")
plt.show()
plt.imshow(img2[:,250,:], origin="lower")
plt.show()
plt.imshow(img2[:,:,700], origin="lower")
plt.show()
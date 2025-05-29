import SimpleITK as sitk

import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib


# Read with SimpleITK
moving_image = sitk.ReadImage('nifti/2/brain2.nii', sitk.sitkFloat64)
# moving_image = sitk.ReadImage('nifti/1/brain1.nii', sitk.sitkFloat64)

# Reference image
fixed_image = sitk.ReadImage('nifti/miplab-ncct_sym_brain.nii.gz', sitk.sitkFloat64)


# plt.imshow(sitk.GetArrayViewFromImage(moving_image)[256], origin="lower")
plt.imshow(sitk.GetArrayViewFromImage(moving_image)[123], origin="lower")
plt.show()
plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[100], origin="lower")
plt.show()


# Must be same type + same size
print(sitk.GetArrayViewFromImage(fixed_image).shape)
print(sitk.GetArrayViewFromImage(moving_image).shape)
# fixed_image = sitk.GetImageFromArray(head.astype(float))


print("pixel type: " + str(fixed_image.GetPixelIDTypeAsString()))
print("pixel type: " + str(moving_image.GetPixelIDTypeAsString()))



# # print(fixed_image, moving_image)
# image_viewer = sitk.ImageViewer()
# image_viewer.SetApplication(r"C:\Users\maryl\Fiji\fiji-windows-x64.exe")
# image_viewer.Execute(fixed_image)


# Optimization only works well with gradients. I can't use binary masks.
initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)


registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10,
)
# registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.
# registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
# registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
# registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Storage list for metric values.
metric_values = []
# Function to record the metric value at each iteration.
def record_metric():
    metric_values.append(registration_method.GetMetricValue())
# Connect our function to be called at each iteration.
registration_method.AddCommand(sitk.sitkIterationEvent, record_metric)


final_transform = registration_method.Execute(fixed_image, moving_image)


# Plot : convergence of the optimizer
plt.figure(figsize=(8, 4))
plt.plot(metric_values, 'r-')
plt.plot(0, metric_values[0], 'b*')  # point de départ
plt.xlabel("Iteration Number")
plt.ylabel("Metric Value")
plt.title("Convergence of the Optimizer")
plt.grid(True)
plt.tight_layout()
plt.show()

# Always check the reason optimization terminated.
print("Final metric value: {0}".format(registration_method.GetMetricValue()))
print(
    "Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()
    )
)
# print(initial_transform)
print(final_transform)


# MARCHE PAS
fixed_index = (100, 300, 300)  # Just an example

# Convert index to physical point
fixed_point_phys = fixed_image.TransformIndexToPhysicalPoint(fixed_index)

# Apply the transformation
transformed_point = final_transform.TransformPoint(fixed_point_phys)

# Optional: Convert the physical point back to index in moving image
moving_index = moving_image.TransformPhysicalPointToIndex(transformed_point)

print(f"Fixed image index: {fixed_index}")
print(f"Mapped physical point on moving image: {transformed_point}")
print(f"Corresponding index on moving image: {moving_index}")

ax1 = plt.subplot(121)
ax1.imshow(sitk.GetArrayViewFromImage(moving_image)[fixed_index[0]], origin="lower") # (z, y, x)
ax1.scatter([fixed_index[1]], [fixed_index[2]], c="r")

ax2 = plt.subplot(122)
ax2.imshow(sitk.GetArrayViewFromImage(fixed_image)[moving_index[0]], origin="lower") # (z, y, x)
ax2.scatter([moving_index[1]], [moving_index[2]], c="r")

plt.show()





# # Assuming you have fixed_image, moving_image, and transform from registration

# resampler = sitk.ResampleImageFilter()
# resampler.SetReferenceImage(fixed_image)
# resampler.SetInterpolator(sitk.sitkLinear)  # Or sitk.sitkNearestNeighbor
# resampler.SetDefaultPixelValue(100)  # Set a default value for pixels outside the image
# resampler.SetTransform(final_transform)

# out_image = resampler.Execute(moving_image)

# import matplotlib.pyplot as plt
# import numpy as np

# def display_images(fixed_image, moving_image, out_image):
#     # Convert SimpleITK images to NumPy arrays
#     fixed_array = sitk.GetArrayFromImage(fixed_image)
#     moving_array = sitk.GetArrayFromImage(moving_image)
#     out_array = sitk.GetArrayFromImage(out_image)

#     # Display the images using matplotlib
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(fixed_array, cmap='gray')
#     plt.title('Fixed Image')
#     plt.subplot(1, 3, 2)
#     plt.imshow(moving_array, cmap='gray')
#     plt.title('Moving Image')
#     plt.subplot(1, 3, 3)
#     plt.imshow(out_array, cmap='gray')
#     plt.title('Registered Image')
#     plt.show()

# display_images(fixed_image, moving_image, out_image)
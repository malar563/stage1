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
registration_method.SetOptimizerScalesFromPhysicalShift()

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
# print(final_transform)

import SimpleITK as sitk

import numpy as np
import os

image_viewer = sitk.ImageViewer()
image_viewer.SetApplication(r"C:\Users\maryl\Fiji\fiji-windows-x64.exe")

OUTPUT_DIR = "output"

fixed_image = sitk.ReadImage('nifti/6_cow_angio__06__hv36__3.nii.gz', sitk.sitkFloat32)
moving_image = sitk.ReadImage('nifti/301_carotid_angio_0625mm.nii.gz', sitk.sitkFloat32)

# print(fixed_image, moving_image)
image_viewer.Execute(fixed_image)




# Ici pas encore testé
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

final_transform = registration_method.Execute(fixed_image, moving_image)

# Always check the reason optimization terminated.
print("Final metric value: {0}".format(registration_method.GetMetricValue()))
print("Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()))



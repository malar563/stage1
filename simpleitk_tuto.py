import SimpleITK as sitk
# import registration_gui as rgui

import numpy as np
import os
import matplotlib.pyplot as plt



def open_pickle(file_name="head"):
    import pickle
    with open(file_name+".pickle", "rb") as f:
        head = pickle.load(f)
        head = np.where(head, 1, 0)
    return head

# À faire : resampling

# Read with SimpleITK
moving_image = sitk.ReadImage('nifti/6_cow_angio__06__hv36__3.nii.gz', sitk.sitkFloat64)[:,:,-256:] #[-256:,:,:] # Marche pas
fixed_image = sitk.ReadImage('nifti/301_carotid_angio_0625mm.nii.gz', sitk.sitkFloat64)[:,:,-256:]



plt.imshow(sitk.GetArrayViewFromImage(moving_image)[2])
plt.show()

# numpy -> SimpleITK
head = open_pickle()
# Must be same type + same size
print(head.shape)
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

import os
import nibabel as nib
import pandas as pd
from class_identification import Identification


def load_landmarks_from_csv(csv_path):
    """Load and separate MRI and CT landmarks from a CSV file."""
    df = pd.read_csv(csv_path, sep=",")
    
    # Get MRI landmarks
    array_MRI = df.iloc[6:12, 1:].values.astype(float)
    reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI = array_MRI[1], array_MRI[3], array_MRI[5]
    imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI = array_MRI[0], array_MRI[2], array_MRI[4]
    
    # Get CT landmarks
    array_CT = df.iloc[13:19, 1:].values.astype(float)
    reg_nas_CT, reg_lpa_CT, reg_rpa_CT = array_CT[1], array_CT[3], array_CT[5]
    imp_nas_CT, imp_lpa_CT, imp_rpa_CT = array_CT[0], array_CT[2], array_CT[4]

    return {"MRI": {"reg": [reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI],
                    "imp": [imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI]},
            "CT": {"reg":  [reg_nas_CT,  reg_lpa_CT,  reg_rpa_CT],
                   "imp":  [imp_nas_CT,  imp_lpa_CT,  imp_rpa_CT]}}


# ------------------ USER SETTINGS ------------------ #
    
# Choose working folder and file number
big_output_directory = "cava" # Folder with NIfTI and CSV files
file_number = 1 # Number of the case to visualize

# Show landmarks in normal CT space (to be transformed in patient space)
show_CT_normalized_space = True
path_CT_not_normalized = 'head1.nii.gz'

show_landmarks = True # To see the landmarks
# Choose which landmarks to display (can be CT or MRI, reg or imp)
reg_with = "CT" # "CT" or "MRI"
landmarks_type = "reg" # "reg" or "imp"

show_file = True # To see a specific file
# Choose which NIfTI image and axis to display (comment/uncomment/change here)
nifti_img_name = "mask" + str(file_number) # e.g., "mask", "totalsegmentator", "mca_territory", "head"
nifti_img_name = "6_cow_angio__06__hv36__3"
axis = 2 # 0:y-axis, 1:x-axis, 2:z-axis

# ---------------- END OF USER SETTINGS ---------------- #

# Initialize an instance of the class
if show_CT_normalized_space:
    id = Identification(big_output_directory=big_output_directory, file_number=file_number, fixed_img_path=path_CT_not_normalized, register_with_CT_not_normalized=True)

# Initialize an instance of the class
id = Identification(big_output_directory=big_output_directory, file_number=file_number, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

# Show head mask
id.show_3D_array(id.head, axis=0)

if show_landmarks:
    csv_path = os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv")
    head = nib.load(id.moving_img_path)
    head = head.get_fdata()

    dict_landmarks = load_landmarks_from_csv(csv_path=csv_path)
    nas, lpa, rpa = dict_landmarks[reg_with][landmarks_type]

    # Show head with landmarks
    id.show_3D_array(head, axis=0, pts=[ ((lpa[2],lpa[0]),lpa[1],"blue"),  ((rpa[2],rpa[0]),rpa[1],"red"), ((nas[2],nas[0]),nas[1],"green")])
    id.show_3D_array(head, axis=1, pts=[ ((lpa[2],lpa[1]),lpa[0],"blue"),  ((rpa[2],rpa[1]),rpa[0],"red"), ((nas[2],nas[1]),nas[0],"green")])
    id.show_3D_array(head, axis=2, pts=[ ((lpa[0],lpa[1]),lpa[2],"blue"),  ((rpa[0],rpa[1]),rpa[2],"red"), ((nas[0],nas[1]),nas[2],"green")])

if show_file:
    # Show a specific file
    img = nib.load(os.path.join(id.nifti_output_directory, nifti_img_name+".nii.gz"))
    img = img.get_fdata()
    id.show_3D_array(img, axis=axis)


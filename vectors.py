import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
import numpy as np
from class_identification import Identification


dict_scans = {
    "250_CQ/CQ500CT0 CQ500CT0/Unknown Study/CT PLAIN THIN" : {"NAS":[450,235,40], "LPA":[253,94,70], "RPA":[281,402,40], "res":[0.451172 , 0.451172 , 0.6252062], "dim":[512., 512., 240.]},        
    "250_CQ/CQ500CT2 CQ500CT2/Unknown Study/CT 0.625mm" : {"NAS":[433,231,75], "LPA":[218,117,60], "RPA":[262,407,84], "res":[0.488281 , 0.488281 , 0.6250661], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT3 CQ500CT3/Unknown Study/CT PLAIN THIN" : {"NAS":[472,265,17], "LPA":[270,87,47], "RPA":[238,389,53], "res":[0.460938  , 0.460938  , 0.62484556], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT4 CQ500CT4/Unknown Study/CT 0.625mm" : {"NAS":[448,268,61], "LPA":[282,122,72], "RPA":[266,383,70], "res":[0.5625    , 0.5625    , 0.62505805], "dim":[512., 512., 288.]},
    "250_CQ/CQ500CT6 CQ500CT6/Unknown Study/CT Thin Details" : {"NAS":[454,255,108], "LPA":[278,101,101], "RPA":[276,391,101], "res":[0.511719, 0.511719, 0.625   ], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT10 CQ500CT10/Unknown Study/CT PLAIN THIN" : {"NAS":[434,274,33], "LPA":[247,119,30], "RPA":[225,406,37], "res":[0.464844  , 0.464844  , 0.62485915], "dim":[512., 512., 224.]},
    "250_CQ/CQ500CT17 CQ500CT17/Unknown Study/CT 0.625mm" : {"NAS":[412,240,90], "LPA":[243,108,54], "RPA":[257,383,48], "res":[0.546875  , 0.546875  , 0.62530327], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT18 CQ500CT18/Unknown Study/CT 0.625mm" : {"NAS":[427,250,106], "LPA":[246,113,62], "RPA":[250,385,67], "res":[0.488281 , 0.488281 , 0.6251072], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT22 CQ500CT22/Unknown Study/CT PLAIN THIN" : {"NAS":[448,252,20], "LPA":[251,65,41], "RPA":[236,395,31], "res":[0.445312, 0.445312, 0.624315], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT26 CQ500CT26/Unknown Study/CT PLAIN THIN" : {"NAS":[429,252,26], "LPA":[254,85,39], "RPA":[232,388,37], "res":[0.412109, 0.412109, 0.625   ], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT40 CQ500CT40/Unknown Study/CT 0.625mm" : {"NAS":[447,262,97], "LPA":[287,113,94], "RPA":[260,367,64], "res":[0.541016  , 0.541016  , 0.62431645], "dim":[512., 512., 288.]},
    "48" : {"NAS":[457,238,109], "LPA":[252,63,90], "RPA":[228,365,48], "res":[], "dim":[]},
    "250_CQ/CQ500CT50 CQ500CT50/Unknown Study/CT 0.625mm" : {"NAS":[449,264,75], "LPA":[276,105,60], "RPA":[277,388,39], "res":[0.525391  , 0.525391  , 0.62495667], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT55 CQ500CT55/Unknown Study/CT 0.625mm" : {"NAS":[458,252,125], "LPA":[271,86,70], "RPA":[262,414,71], "res":[0.488281  , 0.488281  , 0.62495834], "dim":[512., 512., 288.]},
    "57" : {"NAS":[459,202,63], "LPA":[227,96,34], "RPA":[296,382,45], "res":[], "dim":[]},
    "250_CQ/CQ500CT60 CQ500CT60/Unknown Study/CT 0.625mm-2" : {"NAS":[450,249,132], "LPA":[255,104,55], "RPA":[253,403,86], "res":[0.488281 , 0.488281 , 0.6246113], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT66 CQ500CT66/Unknown Study/CT PLAIN THIN" : {"NAS":[433,262,64], "LPA":[255,106,45], "RPA":[226,375,51], "res":[0.488281, 0.488281, 0.625   ], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT67 CQ500CT67/Unknown Study/CT PLAIN THIN" : {"NAS": [457, 263, 49],  "LPA": [253, 85, 48],   "RPA": [235, 398, 48], "res":[0.457031  , 0.457031  , 0.62421215], "dim":[512., 512., 240.]},
    "73" : {"NAS": [418, 250, 18],  "LPA": [223, 94, 29],   "RPA": [215, 388, 24], "res":[], "dim":[]},
    "250_CQ/CQ500CT78 CQ500CT78/Unknown Study/CT PLAIN THIN" : {"NAS": [448, 250, 53],  "LPA": [247, 94, 63],   "RPA": [240, 385, 55], "res":[0.488281, 0.488281, 0.625   ], "dim":[512., 512., 256.]},
    "80" : {"NAS": [408, 234, 110], "LPA": [253, 122, 54],  "RPA": [287, 375, 63], "res":[], "dim":[]},
    "250_CQ/CQ500CT84 CQ500CT84/Unknown Study/CT PLAIN THIN" : {"NAS": [450, 257, 69],  "LPA": [258, 93, 72],   "RPA": [242, 394, 74], "res":[0.488281  , 0.488281  , 0.62505805], "dim":[512., 512., 272.]},
    "250_CQ/CQ500CT85 CQ500CT85/Unknown Study/CT PLAIN THIN" : {"NAS": [432, 246, 110], "LPA": [257, 101, 84],  "RPA": [258, 362, 74], "res":[0.488281, 0.488281, 0.625   ], "dim":[512., 512., 272.]},
    "90" : {"NAS": [417, 235, 88],  "LPA": [224, 87, 149],  "RPA": [237, 378, 63], "res":[], "dim":[]},
    "250_CQ/CQ500CT92 CQ500CT92/Unknown Study/CT PLAIN THIN" : {"NAS": [450, 245, 34],  "LPA": [262, 84, 47],   "RPA": [251, 377, 34], "res":[0.488281 , 0.488281 , 0.6255545], "dim":[512., 512., 256.]},
    "101" : {"NAS": [442, 259, 19],  "LPA": [253, 103, 32],  "RPA": [243, 394, 39], "res":[], "dim":[]},
    "250_CQ/CQ500CT102 CQ500CT102/Unknown Study/CT PLAIN THIN" : {"NAS": [446, 258, 30],  "LPA": [243, 88, 50],   "RPA": [192, 390, 35], "res":[0.453125, 0.453125, 0.625   ], "dim":[512., 512., 240.]},
    "104" : {"NAS": [485, 252, 24],  "LPA": [265, 82, 40],   "RPA": [250, 396, 36], "res":[], "dim":[]},
    "250_CQ/CQ500CT108 CQ500CT108/Unknown Study/CT 0.625mm" : {"NAS": [421, 260, 108], "LPA": [243, 99, 86],   "RPA": [274, 402, 48], "res":[0.496094 , 0.496094 , 0.6257798], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT109 CQ500CT109/Unknown Study/CT 0.625mm" : {"NAS": [449, 243, 67],  "LPA": [235, 86, 62],   "RPA": [242, 396, 46], "res":[0.488281 , 0.488281 , 0.6250672], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT111 CQ500CT111/Unknown Study/CT PLAIN THIN" : {"NAS": [484, 258, 115], "LPA": [266, 62, 75],   "RPA": [253, 409, 67], "res":[0.417969, 0.417969, 0.625   ], "dim":[512., 512., 272.]},
    "250_CQ/CQ500CT113 CQ500CT113/Unknown Study/CT PLAIN THIN" : {"NAS": [438, 266, 62],  "LPA": [240, 94, 63],   "RPA": [220, 399, 71], "res":[0.46875  , 0.46875  , 0.6246113], "dim":[512., 512., 272.]},
    "250_CQ/CQ500CT121 CQ500CT121/Unknown Study/CT PLAIN THIN" : {"NAS": [466, 241, 94],  "LPA": [256, 81, 75],   "RPA": [254, 403, 87], "res":[0.488281  , 0.488281  , 0.62510717], "dim":[512., 512., 288.]},
    "250_CQ/CQ500CT126 CQ500CT126/Unknown Study/CT PLAIN THIN" : {"NAS": [440, 258, 102], "LPA": [253, 97, 48],   "RPA": [256, 420, 66], "res":[0.439453, 0.439453, 0.625   ], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT130 CQ500CT130/Unknown Study/CT PLAIN THIN" : {"NAS": [436, 242, 28],  "LPA": [229, 92, 48],   "RPA": [227, 391, 39], "res":[0.466797 , 0.466797 , 0.6251082], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT135 CQ500CT135/Unknown Study/CT PLAIN THIN" : {"NAS": [455, 266, 43],  "LPA": [254, 104, 21],  "RPA": [244, 414, 27], "res":[0.453125  , 0.453125  , 0.62523353], "dim":[512., 512., 224.]},
    "250_CQ/CQ500CT140 CQ500CT140/Unknown Study/CT PLAIN THIN" : {"NAS": [420, 250, 76],  "LPA": [254, 113, 73],  "RPA": [238, 361, 61], "res":[0.488281 , 0.488281 , 0.6254716], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT149 CQ500CT149/Unknown Study/CT 0.625mm-3" : {"NAS": [442, 271, 53],  "LPA": [279, 96, 77],   "RPA": [226, 380, 53], "res":[0.488281  , 0.488281  , 0.62505805], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT152 CQ500CT152/Unknown Study/CT PLAIN THIN" : {"NAS":[439,277,54], "LPA":[281,91,51], "RPA":[248,404,46], "res":[0.488281 , 0.488281 , 0.6258083], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT154 CQ500CT154/Unknown Study/CT PLAIN THIN" : {"NAS":[464,256,69], "LPA":[258,73,53], "RPA":[243,416,71], "res":[0.441406  , 0.441406  , 0.62431645], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT162 CQ500CT162/Unknown Study/CT PLAIN THIN-2" : {"NAS":[441,230,45], "LPA":[247,81,59], "RPA":[261,394,60], "res":[0.419922 , 0.419922 , 0.6248593], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT166 CQ500CT166/Unknown Study/CT 0.625mm-2" : {"NAS":[455,236,123], "LPA":[291,79,75], "RPA":[287,382,66], "res":[0.488281  , 0.488281  , 0.62555593], "dim":[512., 512., 288.]},
    "250_CQ/CQ500CT167 CQ500CT167/Unknown Study/CT 0.625mm" : {"NAS":[435,231,103], "LPA":[257,100,64], "RPA":[267,380,66], "res":[0.515625  , 0.515625  , 0.62505805], "dim":[512., 512., 288.]},
    "250_CQ/CQ500CT179 CQ500CT179/Unknown Study/CT PLAIN THIN" : {"NAS":[429,265,46], "LPA":[240,99,35], "RPA":[215,396,48], "res":[0.455078  , 0.455078  , 0.62499994], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT183 CQ500CT183/Unknown Study/CT PLAIN THIN" : {"NAS":[441,249,135], "LPA":[259,91,110], "RPA":[237,372,98], "res":[0.488281, 0.488281, 0.625   ], "dim":[512., 512., 300.]},
    "250_CQ/CQ500CT188 CQ500CT188/Unknown Study/CT 0.625mm-3" : {"NAS":[441,250,49], "LPA":[245,106,60], "RPA":[253,386,62], "res":[0.5      , 0.5      , 0.6252071], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT193 CQ500CT193/Unknown Study/CT 0.625mm" : {"NAS":[436,261,66], "LPA":[264,113,64], "RPA":[246,388,65], "res":[0.488281  , 0.488281  , 0.62561053], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT196 CQ500CT196/Unknown Study/CT 0.625mm" : {"NAS":[447,224,99], "LPA":[243,100,64], "RPA":[301,406,81], "res":[0.488281 , 0.488281 , 0.6253051], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT204 CQ500CT204/Unknown Study/CT PLAIN THIN" : {"NAS":[453,251,41], "LPA":[244,78,46], "RPA":[234,407,45], "res":[0.445312  , 0.445312  , 0.62505805], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT216 CQ500CT216/Unknown Study/CT 0.625mm" : {"NAS":[430,240,72], "LPA":[240,123,50], "RPA":[263,399,64], "res":[0.53125  , 0.53125  , 0.6244705], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT219 CQ500CT219/Unknown Study/CT 0.625mm" : {"NAS":[454,244,143], "LPA":[270,93,68], "RPA":[289,415,79], "res":[0.488281  , 0.488281  , 0.62496215], "dim":[512., 512., 288.]},
    "250_CQ/CQ500CT220 CQ500CT220/Unknown Study/CT PLAIN THIN" : {"NAS":[462,265,80], "LPA":[272,99,74], "RPA":[241,386,81], "res":[0.488281  , 0.488281  , 0.62485963], "dim":[512., 512., 272.]},
    "250_CQ/CQ500CT221 CQ500CT221/Unknown Study/CT PLAIN THIN" : {"NAS":[456,235,40], "LPA":[253,96,42], "RPA":[253,389,47], "res":[0.488281  , 0.488281  , 0.62523365], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT233 CQ500CT233/Unknown Study/CT PLAIN THIN" : {"NAS":[462,258,38], "LPA":[252,80,38], "RPA":[221,393,44], "res":[0.458984, 0.458984, 0.625   ], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT237 CQ500CT237/Unknown Study/CT PLAIN THIN" : {"NAS":[435,247,63], "LPA":[240,97,49], "RPA":[232,374,46], "res":[0.488281, 0.488281, 0.625   ], "dim":[512., 512., 240.]},
    "250_CQ/CQ500CT241 CQ500CT241/Unknown Study/CT PLAIN THIN-2" : {"NAS":[402,269,169], "LPA":[231,115,120], "RPA":[221,380,110], "res":[0.546875, 0.546875, 0.625   ], "dim":[512., 512., 300.]}, # bad scan
    "250_CQ/CQ500CT243 CQ500CT243/Unknown Study/CT PLAIN THIN" : {"NAS":[448,241,76], "LPA":[247,79,47], "RPA":[252,395,39], "res":[0.482422 , 0.482422 , 0.6251091], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT249 CQ500CT249/Unknown Study/CT 0.625mm" : {"NAS":[423,272,93], "LPA":[263,121,48], "RPA":[240,393,61], "res":[0.507812 , 0.507812 , 0.6253051], "dim":[512., 512., 256.]},
    "250_CQ/CQ500CT250 CQ500CT250/Unknown Study/CT PLAIN THIN" : {"NAS":[452,249,21], "LPA":[257,105,34], "RPA":[252,385,34], "res":[0.488281 , 0.488281 , 0.6242131], "dim":[512., 512., 240.]},
}




def load_landmarks_from_csv(csv_path):
    """Load and separate MRI and CT landmarks from a CSV file."""
    df = pd.read_csv(csv_path, sep=",", header=None)
    print(df.iloc[1,0])
    
    # Get MRI landmarks
    scan_name = df.iloc[1,0]
    array_MRI = df.iloc[6:12, 1:].values.astype(float)
    reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI = array_MRI[1], array_MRI[3], array_MRI[5]
    imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI = array_MRI[0], array_MRI[2], array_MRI[4]
    
    # Get CT landmarks
    array_CT = df.iloc[12:18, 1:].values.astype(float)
    reg_nas_CT, reg_lpa_CT, reg_rpa_CT = array_CT[1], array_CT[3], array_CT[5]
    imp_nas_CT, imp_lpa_CT, imp_rpa_CT = array_CT[0], array_CT[2], array_CT[4]

    return {"MRI": {"reg": [reg_nas_MRI, reg_lpa_MRI, reg_rpa_MRI],
                    "imp": [imp_nas_MRI, imp_lpa_MRI, imp_rpa_MRI]},
            "CT": {"reg":  [reg_nas_CT,  reg_lpa_CT,  reg_rpa_CT],
                   "imp":  [imp_nas_CT,  imp_lpa_CT,  imp_rpa_CT]},
            "scan name":scan_name}


# ------------------ USER SETTINGS ------------------ #
    
# Choose working folder and file number
big_output_directory = "250_2025-07-31" # Folder with NIfTI and CSV files


# Show landmarks in normal CT space (to be transformed in patient space)
show_CT_normalized_space = False
path_CT_not_normalized = 'head1.nii.gz'

show_landmarks = False # To see the landmarks
show_my = False
# Choose which landmarks to display (can be CT or MRI, reg or imp)
reg_with = "MRI" # "CT" or "MRI"
landmarks_type = "reg" # "reg" or "imp"


# ---------------- END OF USER SETTINGS ---------------- #

dict_nas = {"x_origins" : [],"y_origins": [],"z_origins" : [],"u_comps" : [],"v_comps" : [],"w_comps" : []}
dict_lpa = {"x_origins" : [],"y_origins": [],"z_origins" : [],"u_comps" : [],"v_comps" : [],"w_comps" : []}
dict_rpa = {"x_origins" : [],"y_origins": [],"z_origins" : [],"u_comps" : [],"v_comps" : [],"w_comps" : []}
for file_number in range(4):
    # Initialize an instance of the class
    id = Identification(big_output_directory=big_output_directory, file_number=file_number, fixed_img_path='icbm_avg_152_t1_tal_lin.nii')

    csv_path = os.path.join(id.nifti_output_directory, "points"+id.file_number+".csv")
    head = nib.load(id.moving_img_path)
    head = head.get_fdata()

    dict_landmarks = load_landmarks_from_csv(csv_path=csv_path)
    scan_name = dict_landmarks["scan name"]
    res = np.array(dict_scans[scan_name]["res"])
    nas, lpa, rpa = dict_landmarks[reg_with][landmarks_type]
    nas = np.array(nas)*res
    lpa = np.array(lpa)*res
    rpa = np.array(rpa)*res
    
    
    my_nas = np.array(dict_scans[scan_name]["NAS"])*res
    my_lpa = np.array(dict_scans[scan_name]["LPA"])*res
    my_rpa = np.array(dict_scans[scan_name]["RPA"])*res

    points = [nas, lpa, rpa]
    my_points = [my_nas, my_lpa, my_rpa]
    all_dict = [dict_nas, dict_lpa, dict_rpa]
    for i in range(3):
        x, y, z = points[i]
        u, v, w = np.array(my_points[i]) - np.array(points[i]) 
        all_dict[i]["x_origins"].append(x)
        all_dict[i]["y_origins"].append(y)
        all_dict[i]["z_origins"].append(z)
        all_dict[i]["u_comps"].append(u)
        all_dict[i]["v_comps"].append(v)
        all_dict[i]["w_comps"].append(w)

all_dict = [dict_nas, dict_lpa, dict_rpa]
colors = ["green", "blue", "red"]
w_tot = []
u_tot = []
for i in range(3):
    x_origins=all_dict[i]["x_origins"]
    y_origins=all_dict[i]["y_origins"]
    z_origins=all_dict[i]["z_origins"]
    u_comps=all_dict[i]["u_comps"]
    v_comps=all_dict[i]["v_comps"]
    w_comps=all_dict[i]["w_comps"]   
    plt.quiver(z_origins, x_origins, w_comps, u_comps,
            angles='xy', scale_units='xy', scale=1,
            color=colors[i])

    # Set plot limits for better visualization
    plt.xlim(0, 511)
    plt.ylim(0, 511)

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.title("50 Single Vectors")

    # Display the plot
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # Ensure equal scaling for x and y axes
    plt.show()

    w_tot.append(np.sum(w_comps))
    u_tot.append(np.sum(u_comps))


plt.quiver([0,0,0], [0,0,0], w_tot, u_tot,
            angles='xy', scale_units='xy', scale=1,
            color=["green", "blue", "red"])

# Set plot limits for better visualization
plt.xlim(-200, 200)
plt.ylim(-200, 200)

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.title("50 Single Vectors")

# Display the plot
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box') # Ensure equal scaling for x and y axes
plt.show()



    # if show_landmarks:
        

    #     # Show head with landmarks
    #     id.show_3D_array(head, axis=0, pts=[ ((lpa[2],lpa[0]),lpa[1],"blue"),  ((rpa[2],rpa[0]),rpa[1],"red"), ((nas[2],nas[0]),nas[1],"green")])
    #     id.show_3D_array(head, axis=1, pts=[ ((lpa[2],lpa[1]),lpa[0],"blue"),  ((rpa[2],rpa[1]),rpa[0],"red"), ((nas[2],nas[1]),nas[0],"green")])
    #     id.show_3D_array(head, axis=2, pts=[ ((lpa[0],lpa[1]),lpa[2],"blue"),  ((rpa[0],rpa[1]),rpa[2],"red"), ((nas[0],nas[1]),nas[2],"green")])


    # # Vector 1: starts at (0,0), points towards (2,3)
    # x1, y1 = 0, 0
    # u1, v1 = 2, 3

    # # Vector 2: starts at (1,1), points towards (4,-1)
    # x2, y2 = 1, 1
    # u2, v2 = 3, -2  # (4-1, -1-1)

    # # Vector 3: starts at (-2,0), points towards (0,2)
    # x3, y3 = -2, 0
    # u3, v3 = 2, 2


    
    # plt.quiver(x2, y2, u2, v2, color='blue', angles='xy', scale_units='xy', scale=1, label='Vector 2')
    # plt.quiver(x3, y3, u3, v3, color='green', angles='xy', scale_units='xy', scale=1, label='Vector 3')



    # plt.xlim(-3, 5)
    # plt.ylim(-3, 4)
    # plt.axhline(0, color='grey', lw=0.5)
    # plt.axvline(0, color='grey', lw=0.5)
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Multiple Single Vectors')
    # plt.grid(True)
    # plt.legend() # If using individual calls with labels
    # plt.gca().set_aspect('equal', adjustable='box') # Ensures correct aspect ratio for vectors
    # plt.show()




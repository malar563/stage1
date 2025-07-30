import pandas as pd
import numpy as np
import os
from automatically_get_dicom_folders import create_list
import matplotlib.pyplot as plt


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


def compute_distance(pt1, pt2, res, remove_dim=None):
    
    if len(pt1) > 0 and len(pt2) > 0:
        if remove_dim == "x":
            pt1[0], pt2[0] = 0, 0
        elif remove_dim == "y":
            pt1[1], pt2[1] = 0, 0
        elif remove_dim == "z":
            pt1[2], pt2[2] = 0, 0
        elif remove_dim is not None:
            return None
        distance = np.sqrt(((res[0]*(pt1[0]-pt2[0]))**2 + (res[1]*(pt1[1]-pt2[1]))**2 + (res[2]*(pt1[1]-pt2[1]))**2))
        return distance
    return None


def compute_angle(u, v, norm_u, norm_v):
    angle = np.arccos((np.dot(u,v)/(norm_u*norm_v)))
    return angle*180/np.pi



def make_dict_scans(list_csv_paths):

    dict_scans = {}

    for csv_path in list_csv_paths:
        csv_file = [f for f in os.listdir(csv_path) if f.endswith(".csv")]

        csv_file_path = os.path.join(csv_path, csv_file[0])
        df = pd.read_csv(csv_file_path, sep=",", header=None, on_bad_lines='skip')

        reg_type = {"_MRI_imp":[6,8,10], "_MRI_reg":[7,9,11], "_CT_imp":[12,14,16], "_CT_reg":[13,15,17]}
        for subtype in reg_type:
            index = reg_type[subtype]
            dict_scans[df.iloc[1,0]+subtype] = {"NAS":df.iloc[index[0],1:].values.astype(float), 
                                                "LPA":df.iloc[index[1],1:].values.astype(float), 
                                                "RPA":df.iloc[index[2],1:].values.astype(float), 
                                                "res":df.iloc[4,1:].values.astype(float), 
                                                "dim":df.iloc[2,1:].values.astype(float),
                                                "folder_path":csv_path}
    return dict_scans





def find_caracteristics(dict_scans):
    dict_props = {}
    for scan in dict_scans:
        points = dict_scans[scan]
        if len(points["res"]) > 0:
            dict_props.setdefault('scan name', []).append(scan)
            dict_props.setdefault('dim', []).append(points["dim"])
            dict_props.setdefault('NAS', []).append(points["NAS"])
            dict_props.setdefault('LPA', []).append(points["LPA"])
            dict_props.setdefault('RPA', []).append(points["RPA"])
            vec_nas_lpa = (np.array(points["NAS"]) - np.array(points["LPA"]))*np.array(points["res"])
            vec_nas_rpa = (np.array(points["NAS"]) - np.array(points["RPA"]))*np.array(points["res"])
            vec_lpa_rpa = (np.array(points["LPA"]) - np.array(points["RPA"]))*np.array(points["res"])
            d_nas_lpa = compute_distance(points["NAS"], points["LPA"], points["res"])
            d_nas_rpa = compute_distance(points["NAS"], points["RPA"], points["res"])
            d_lpa_rpa = compute_distance(points["LPA"], points["RPA"], points["res"])
            area_triangle = 0.25*np.sqrt((d_nas_lpa**2 + d_nas_rpa**2 + d_lpa_rpa**2)**2 - (2*(d_nas_lpa**4 + d_nas_rpa**4 + d_lpa_rpa**4))) # Heron's formula
            dict_props.setdefault('distance NAS LPA', []).append(d_nas_lpa)
            dict_props.setdefault('distance NAS RPA', []).append(d_nas_rpa)
            dict_props.setdefault('distance LPA RPA', []).append(d_lpa_rpa)
            dict_props.setdefault('area triangle', []).append(area_triangle)

            mid_lpa_rpa_pt = (np.array(points["LPA"]) + np.array(points["RPA"]))*np.array(points["res"])/2
            vec_nas_mid = (np.array(points["NAS"])*np.array(points["res"])) - mid_lpa_rpa_pt
            d_nas_mid_pt = compute_distance((np.array(points["NAS"])*np.array(points["res"])), mid_lpa_rpa_pt, [1,1,1])

            angle = compute_angle(vec_lpa_rpa,vec_nas_mid, d_lpa_rpa, d_nas_mid_pt)
            dict_props.setdefault("angle between axis", []).append(angle)

            angle_lpa_nas_rpa = compute_angle(vec_nas_lpa, vec_nas_rpa, d_nas_lpa, d_nas_rpa)
    return dict_props



def cut(dict_props2check, dict_reference):
    dict_out_of_criteria = {}

    # Distance NAS-LPA and NAS-RPA should be approx. equal
    d_nas_lpa = dict_props2check['distance NAS LPA'] 
    d_nas_rpa = dict_props2check['distance NAS RPA']
    plt.hist(np.array(d_nas_lpa)-np.array(d_nas_rpa), bins=30)
    plt.xlabel("Différence entre la distance LPA-NAS et RPA-NAS")
    plt.ylabel("Nombre de scans")
    plt.show()

    # Mean surface triangle
    mean = np.mean(dict_reference['area triangle'])
    std = np.std(dict_reference['area triangle'])
    print("Average surface triangle :", mean, "±", 3*std)
    plt.hist(dict_props2check['area triangle'], bins=15)
    plt.axvline(mean+std, color='r', linestyle='--')
    plt.axvline(mean-std, color='r', linestyle='--')
    plt.axvline(mean+(2*std), color='orange', linestyle='--')
    plt.axvline(mean-(2*std), color='orange', linestyle='--')
    plt.axvline(mean+(3*std), color='yellow', linestyle='--')
    plt.axvline(mean-(3*std), color='yellow', linestyle='--')
    plt.xlabel("Aire de la surface du triangle")
    plt.ylabel("Nombre de scans")
    plt.show()
    for i, surface in enumerate(dict_props2check['area triangle']):
        if surface < mean-(3*std) or surface > mean+(3*std):
            dict_out_of_criteria.setdefault('area triangle', []).append(dict_props2check['scan name'][i])

    # Mean angle axis 
    mean = np.mean(dict_reference['angle between axis'])
    std = np.std(dict_reference['angle between axis'])
    print("Average angle between axis :", mean, "±", 3*std)
    plt.hist(dict_props2check['angle between axis'], bins=15)
    plt.axvline(mean+std, color='r', linestyle='--')
    plt.axvline(mean-std, color='r', linestyle='--')
    plt.axvline(mean+(2*std), color='orange', linestyle='--')
    plt.axvline(mean-(2*std), color='orange', linestyle='--')
    plt.axvline(mean+(3*std), color='yellow', linestyle='--')
    plt.axvline(mean-(3*std), color='yellow', linestyle='--')
    plt.xlabel("Angle formé par l'axe LPA - RPA et l'axe point milieu de LPA/RPA - NAS (deg)")
    plt.ylabel("Nombre de scans")
    plt.show()
    for i, angle in enumerate(dict_props2check['angle between axis']):
        if angle < mean-(3*std) or angle > mean+(3*std):
            dict_out_of_criteria.setdefault('angle between axis', []).append(dict_props2check['scan name'][i])
    
    diff_z_2check = []
    diff_z_ref = []
    for i, name in enumerate(dict_reference['scan name']):
        ref_lpa = dict_reference['LPA'][i]
        ref_rpa = dict_reference['RPA'][i]
        diff_z_ref.append(ref_lpa[2]-ref_rpa[2])
    
    for i, name in enumerate(dict_props2check['scan name']):
        nas = dict_props2check['NAS'][i]
        lpa = dict_props2check['LPA'][i]
        rpa = dict_props2check['RPA'][i]
        dim = dict_props2check['dim'][i]
        a = dict_props2check['distance NAS LPA'][i] 
        b = dict_props2check['distance NAS RPA'][i]
        c = dict_props2check['distance LPA RPA'][i]
        # One side of the triangle shouldn't be bigger than than the sum of the other two
        if not (a + b > c) and (a + c > b) and (b + c > a):
            dict_out_of_criteria.setdefault('side too long', []).append(dict_props2check['scan name'][i])
        # Nasion is higher than LPA and RPA
        if not (lpa[2] < nas[2]) and (rpa[2] < nas[2]):
            dict_out_of_criteria.setdefault('nasion lower', []).append(dict_props2check['scan name'][i])
        diff_z_2check.append(lpa[2]-rpa[2])
        # No negative number
        if not any(number < 0 for number in nas) and any(number < 0 for number in lpa) and any(number < 0 for number in rpa):
            dict_out_of_criteria.setdefault('negative voxel', []).append(dict_props2check['scan name'][i])
        # Not too close from the edges of the volume
        if any(p[i] < 0.01 * dim[i] or p[i] > 0.99 * dim[i] for i in range(3) for p in [nas, lpa, rpa]):
            dict_out_of_criteria.setdefault('voxel on the edge', []).append(dict_props2check['scan name'][i])
    
    # Mean angle axis 
    mean = np.mean(diff_z_ref)
    std = np.std(diff_z_ref)
    print("Average height difference between ears :", mean, "±", 3*std)
    plt.hist(diff_z_2check, bins=15)
    plt.axvline(mean+std, color='r', linestyle='--')
    plt.axvline(mean-std, color='r', linestyle='--')
    plt.axvline(mean+(2*std), color='orange', linestyle='--')
    plt.axvline(mean-(2*std), color='orange', linestyle='--')
    plt.axvline(mean+(3*std), color='yellow', linestyle='--')
    plt.axvline(mean-(3*std), color='yellow', linestyle='--')
    plt.xlabel("Différence de hauteur entre les deux oreilles")
    plt.ylabel("Nombre de scans")
    plt.show()
    for i, diff in enumerate(diff_z_2check):
        if diff < mean-(3*std) or diff > mean+(3*std):
            dict_out_of_criteria.setdefault('diff z', []).append(dict_props2check['scan name'][i])

    print(dict_out_of_criteria)
    


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# To compare 

# Rename compare identification
def errors_scans(csv_dirs, remove_dim = False, too_far=15):
    dict_distance = {}
    dict_too_far = {}

    for csv_dir in csv_dirs:
        csv_file = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        if csv_file:
            csv_file_path = os.path.join(csv_dir, csv_file[0])
            df = pd.read_csv(csv_file_path, sep=",", header=None, on_bad_lines='skip')
            label = df.iloc[1,0]
            res = df.iloc[4,1:].values.astype(float)
            dim = df.iloc[2,1:].values.astype(float)
            dict_landmarks = dict_scans[label]
            for i in range(6,18):
                pt1 = df.iloc[i,1:].values.astype(float)
                name_pt = df.iloc[i,0]
                if i in [6,7,12,13]:
                    distance = compute_distance(pt1, dict_landmarks["NAS"], res, None)
                    if remove_dim:
                        distance = compute_distance(pt1, dict_landmarks["NAS"], res, "x")
                elif i in [8,9,14,15]:
                    distance = compute_distance(pt1, dict_landmarks["LPA"], res, None)
                    if remove_dim:
                        distance = compute_distance(pt1, dict_landmarks["LPA"], res, "y")
                else:
                    distance = compute_distance(pt1, dict_landmarks["RPA"], res, None)
                    if remove_dim:
                        distance = compute_distance(pt1, dict_landmarks["RPA"], res, "y")
                if distance is None:
                    continue
                dict_distance.setdefault(name_pt, []).append(distance)
                if distance > too_far:
                    dict_too_far[label]= (i, distance)
    return dict_distance, dict_too_far

    




def show_boxplots(dict_errors):

    nas = [dict_errors['MRI Nasion improved (voxel)'], dict_errors['MRI Nasion registered (voxel)'],
           dict_errors['CT Nasion improved (voxel)'], dict_errors['CT Nasion registered (voxel)']]
    lpa = [dict_errors['MRI LPA improved (voxel)'], dict_errors['MRI LPA registered (voxel)'],
           dict_errors['CT LPA improved (voxel)'], dict_errors['CT LPA registered (voxel)']]    
    rpa = [dict_errors['MRI RPA improved (voxel)'], dict_errors['MRI RPA registered (voxel)'],
           dict_errors['CT RPA improved (voxel)'], dict_errors['CT RPA registered (voxel)']]

    landmarks = {"Nasion":nas, "LPA":lpa, "RPA":rpa}
    labels = ["MRI improved", "MRI registered", "CT improved", "CT registered"]
    colors = ["lightcoral", "indianred", "lightsteelblue", "cornflowerblue"]
    medianprops = dict(color='gold')

    for pt_name in landmarks:

        fig, ax = plt.subplots()
        ax.set_ylabel('Distance (mm)')
        ax.set_xlabel(pt_name)
        
        bplot = ax.boxplot(landmarks[pt_name],
                        patch_artist=True,  # fill with color
                        tick_labels=labels,  # will be used to label x-ticks
                        notch=False,
                        medianprops=medianprops)
        # fill with colors
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        plt.show()


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Path of the processing directory (to change)
    directory = "250_2025-07-25" 
    # Create a folder list to get points{...}.csv
    csv_paths = create_list(directory=directory)
    csv_paths.sort(key=lambda x: int(x.split('\\')[-1]))
    # print(csv_paths)

    dict_reg_scans = make_dict_scans(csv_paths)



    dict_props2check = find_caracteristics(dict_reg_scans)
    dict_ref = find_caracteristics(dict_scans)
    cut(dict_props2check, dict_ref)
    # cut(dict_ref, dict_ref)

    # Path of the processing directory (to change)
    directory = "250_2025-07-25" 
    # Create a folder list to get points{...}.csv
    csv_dirs = create_list(directory=directory)
    csv_dirs.sort(key=lambda x: int(x.split('\\')[-1]))

    dict_errors, dict_too_far= errors_scans(csv_dirs=csv_dirs, remove_dim=True, too_far=100)


    # print(dict_errors)

    # show_boxplots(dict_errors=dict_errors)


    # print(dict_too_far)
    # print(len(dict_too_far))

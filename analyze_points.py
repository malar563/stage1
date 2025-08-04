import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import json

from automatically_get_dicom_folders import create_list

# ----------------------------------------------------------------------------

def compute_distance(pt1, pt2, res, remove_dim=None):
    """
    Compute the Euclidean distance between two 3D points, optionally ignoring one axis.

    Parameters
    ----------
    pt1, pt2 : array-like of length >=3
        3D point coordinates.
    res : sequence of 3 floats
        Resolution scaling factors for (x, y, z); differences are multiplied by these before squaring.
    remove_dim : {'x', 'y', 'z'} or None
        If given, zeroes out that dimension in both points before computing distance.

    Returns
    -------
    float or None
        Scaled distance between the two points, or None if inputs are invalid or `remove_dim` is unrecognized.
    """
    if len(pt1) > 0 and len(pt2) > 0:
        if remove_dim == "x":
            pt1[0], pt2[0] = 0, 0
        elif remove_dim == "y":
            pt1[1], pt2[1] = 0, 0
        elif remove_dim == "z":
            pt1[2], pt2[2] = 0, 0
        elif remove_dim is not None:
            return None
        distance = np.sqrt(((res[0]*(pt1[0]-pt2[0]))**2 + 
                            (res[1]*(pt1[1]-pt2[1]))**2 + 
                            (res[2]*(pt1[1]-pt2[1]))**2))
        return distance
    return None


def compute_components(pt1, pt2, res, remove_dim=None):
    """
    Compute the per-axis scaled difference vector between two 3D points, optionally zeroing one axis.

    Parameters
    ----------
    pt1, pt2 : array-like of length >=3
        Input 3D coordinates. Note: if `remove_dim` is set, the corresponding component in both
        points is zeroed in place before computing.
    res : sequence of 3 floats
        Scaling factors for (x, y, z) components.
    remove_dim : {'x', 'y', 'z'} or None
        If given, that dimension is ignored (set to zero).

    Returns
    -------
    tuple of 3 floats or None
        Scaled component-wise differences (pt2 - pt1) * res, or None if inputs are invalid or
        `remove_dim` is unrecognized.
    """
    if len(pt1) > 0 and len(pt2) > 0:
        if remove_dim == "x":
            pt1[0], pt2[0] = 0, 0
        elif remove_dim == "y":
            pt1[1], pt2[1] = 0, 0
        elif remove_dim == "z":
            pt1[2], pt2[2] = 0, 0
        elif remove_dim is not None:
            return None
        return (pt2[0]-pt1[0])*res[0], (pt2[1]-pt1[1])*res[1], (pt2[2]-pt1[2])*res[2]
    return None


def compute_angle(u, v, norm_u, norm_v):
    """
    Compute the angle in degrees between two vectors given their norms.

    Parameters
    ----------
    u, v : array-like
        Input vectors.
    norm_u, norm_v : float
        Precomputed norms (magnitudes) of vectors u and v.

    Returns
    -------
    float
        Angle between u and v in degrees.
    """
    angle = np.arccos((np.dot(u,v)/(norm_u*norm_v)))
    return angle*180/np.pi


def make_dict_scans(list_csv_paths):
    """
    Create a dictionary of scan data extracted from multiple CSV files.

    Parameters
    ----------
    list_csv_paths : list of str
        List of folder paths containing exactly one CSV file each (points{...}.csv).

    Returns
    -------
    dict
        Dictionary where keys are scan identifiers (paths) with subtype suffixes
        ("_MRI_imp", "_MRI_reg", "_CT_imp", "_CT_reg"), and values are dictionaries containing:
            - "NAS", "LPA", "RPA" : ndarray
                Landmark coordinates (voxel).
            - "res" : ndarray
                Voxel resolution.
            - "dim" : ndarray
                Scan dimensions.
            - "folder_path" : str
                Path to the folder containing the CSV file.
        Example of return :
            {"250_CQ/CQ500CT0/CT PLAIN THIN_MRI_reg" : {"NAS":[450,235,40], "LPA":[253,94,70], "RPA":[281,402,40], "res":[0.4, 0.4, 0.6], "dim":[512, 512, 240]},        
            "250_CQ/CQ500CT2/CT 0.625mm_CT_imp"" : {"NAS":[433,231,75], "LPA":[218,117,60], "RPA":[262,407,84], "res":[0.5, 0.5, 0.6], "dim":[512, 512, 256]}}
    """

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

# ----------------------------------------------------------------------------

def find_caracteristics(dict_scans):
    """
    Extract geometric properties from landmarks.

    For each scan with valid resolution, computes and collects:
      - Original landmark positions (NAS, LPA, RPA)
      - Pairwise distances between NAS, LPA, RPA
      - Triangle area formed by those three points
      - Angle between the LPA–RPA axis and the vector from its midpoint to NAS

    Parameters
    ----------
    dict_scans : dict
        Mapping of scan names to landmark data. Each value must contain "NAS", "LPA", "RPA",
        "res", and "dim".

    Returns
    -------
    defaultdict(list)
        Aggregated properties per scan, including distances, area, angle, and raw landmarks.
    """
    dict_props = defaultdict(list)
    for scan_name, points in dict_scans.items():
        if len(points["res"]) > 0:
            dict_props['scan name'].append(scan_name)
            dict_props['dim'].append(points["dim"])
            dict_props['NAS'].append(points["NAS"])
            dict_props['LPA'].append(points["LPA"])
            dict_props['RPA'].append(points["RPA"])

            vec_nas_lpa = (np.array(points["NAS"]) - np.array(points["LPA"]))*np.array(points["res"])
            vec_nas_rpa = (np.array(points["NAS"]) - np.array(points["RPA"]))*np.array(points["res"])
            vec_lpa_rpa = (np.array(points["LPA"]) - np.array(points["RPA"]))*np.array(points["res"])
            d_nas_lpa = compute_distance(points["NAS"], points["LPA"], points["res"])
            d_nas_rpa = compute_distance(points["NAS"], points["RPA"], points["res"])
            d_lpa_rpa = compute_distance(points["LPA"], points["RPA"], points["res"])
            area_triangle = 0.25*np.sqrt((d_nas_lpa**2 + d_nas_rpa**2 + d_lpa_rpa**2)**2 - (2*(d_nas_lpa**4 + d_nas_rpa**4 + d_lpa_rpa**4))) # Heron's formula
            dict_props['distance NAS LPA'].append(d_nas_lpa)
            dict_props['distance NAS RPA'].append(d_nas_rpa)
            dict_props['distance LPA RPA'].append(d_lpa_rpa)
            dict_props['area triangle'].append(area_triangle)

            mid_lpa_rpa_pt = (np.array(points["LPA"]) + np.array(points["RPA"]))*np.array(points["res"])/2
            vec_nas_mid = (np.array(points["NAS"])*np.array(points["res"])) - mid_lpa_rpa_pt
            d_nas_mid_pt = compute_distance((np.array(points["NAS"])*np.array(points["res"])), mid_lpa_rpa_pt, [1,1,1])

            angle = compute_angle(vec_lpa_rpa,vec_nas_mid, d_lpa_rpa, d_nas_mid_pt)
            dict_props["angle between axis"].append(angle)

    return dict_props


def create_histogram(prop_list, label_x):
    """
    Plot a histogram of a property, mark ±1,2,3 standard deviations, and print summary.

    Parameters
    ----------
    prop_list : sequence of numbers
        Values to histogram.
    label_x : str
        X-axis label and name used in the printed summary.

    Returns
    -------
    tuple (mean, std)
        Mean and standard deviation of the input list.
    """
    mean = np.mean(prop_list)
    std = np.std(prop_list)
    print(f"{label_x} :", mean, "±", 3*std)
    plt.hist(prop_list, bins=15)
    plt.axvline(mean+std, color='r', linestyle='--')
    plt.axvline(mean-std, color='r', linestyle='--')
    plt.axvline(mean+(2*std), color='orange', linestyle='--')
    plt.axvline(mean-(2*std), color='orange', linestyle='--')
    plt.axvline(mean+(3*std), color='yellow', linestyle='--')
    plt.axvline(mean-(3*std), color='yellow', linestyle='--')
    plt.xlabel(label_x)
    plt.ylabel("Number of scans")
    plt.show()

    return mean, std


def cut_criteria(dict_props2check, dict_reference, std_criteria=3):
    """
    Evaluate scan quality by comparing geometric landmarks to a reference (or itself) and record failures.

    Flags scans that fall outside tolerances on:
      - Triangle area (± std_criteria standard deviations)
      - Axis angle (± std_criteria standard deviations)
      - Triangle inequality violations ("side too long" : True or False)
      - Nasion's position relative to LPA/RPA ("nasion lower" : True or False) 
      - Negative coordinate values ("negative voxel" : True or False)
      - Landmarks too close to volume edges ("voxel on the edge" : True or False)
      - Height difference between ears outside reference bounds ("diff z" : ± std_criteria standard deviations)

    Writes a CSV "out_of_criteria.csv" summarizing scan names that failed each check.

    Parameters
    ----------
    dict_props2check : dict
        Properties of scans to evaluate (includes distances, angles, landmarks, dims, etc.).
    dict_reference : dict
        Reference distributions used for statistical comparisons.
    std_criteria : float, optional
        Number of standard deviations for tolerance thresholds. Default is 3.
    """
    dict_out_of_criteria = defaultdict(list)

    # Distance NAS-LPA and NAS-RPA should be approx. equal
    d_nas_lpa = dict_props2check['distance NAS LPA'] 
    d_nas_rpa = dict_props2check['distance NAS RPA']
    plt.hist(np.array(d_nas_lpa)-np.array(d_nas_rpa), bins=30)
    plt.xlabel("Difference between LPA-NAS and RPA-NAS distance")
    plt.ylabel("Number of scans")
    plt.show()

    # Mean surface triangle
    mean, std = create_histogram(dict_reference['area triangle'], 'area triangle')
    for i, surface in enumerate(dict_props2check['area triangle']):
        if surface < mean-(std_criteria*std) or surface > mean+(std_criteria*std):
            dict_out_of_criteria['area triangle'].append(dict_props2check['scan name'][i])

    # Mean angle axis LPA-RPA and middle point-NAS
    mean, std = create_histogram(dict_reference['angle between axis'], 'angle between axis')
    for i, angle in enumerate(dict_props2check['angle between axis']):
        if angle < mean-(std_criteria*std) or angle > mean+(std_criteria*std):
            dict_out_of_criteria['angle between axis'].append(dict_props2check['scan name'][i])
    
    diff_z_2check = []
    diff_z_ref = []
    for i, name in enumerate(dict_reference['scan name']):
        ref_lpa, ref_rpa = dict_reference['LPA'][i], dict_reference['RPA'][i]
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
            dict_out_of_criteria['side too long'].append(dict_props2check['scan name'][i])
        # Nasion is higher than LPA and RPA
        if not (lpa[2] < nas[2]) and (rpa[2] < nas[2]):
            dict_out_of_criteria['nasion lower'].append(dict_props2check['scan name'][i])
        diff_z_2check.append(lpa[2]-rpa[2])
        # No negative number
        if not any(number < 0 for number in nas) and any(number < 0 for number in lpa) and any(number < 0 for number in rpa):
            dict_out_of_criteria['negative voxel'].append(dict_props2check['scan name'][i])
        # Not too close from the edges of the volume
        if any(p[i] < 0.01 * dim[i] or p[i] > 0.99 * dim[i] for i in range(3) for p in [nas, lpa, rpa]):
            dict_out_of_criteria['voxel on the edge'].append(dict_props2check['scan name'][i])
    
    # Mean angle axis 
    mean, std = create_histogram(diff_z_ref, 'height difference between ears')
    for i, diff in enumerate(diff_z_2check):
        if diff < mean-(std_criteria*std) or diff > mean+(std_criteria*std):
            dict_out_of_criteria['LPA-RPA diff z'].append(dict_props2check['scan name'][i])

    import pandas as pd
    df = pd.concat({k: pd.Series(v) for k, v in dict_out_of_criteria.items()}, axis=1)
    # After concat the columns are a MultiIndex (key, ), so flatten:
    df.columns = list(dict_out_of_criteria.keys())
    df.to_csv("out_of_criteria.csv", index=False)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------ USER SETTINGS ------------------ #
    """
    User-configurable settings :
        - directory: folder containing subfolders with points{file_number}.csv to analyze.
        - std_criteria: number of standard deviations used as tolerance when comparing geometric 
            metrics (e.g., triangle area, angle) to a reference distribution.
        - reference_distribution_for_std_cut: if True, uses the external reference (250_CQ.json) 
            to define the standard deviation cutoffs; if False, uses the current data as its own reference.

    Outputs out_of_criteria.csv :
        - area triangle : Surface of the triangle formed with the 3 landmarks
        - angle between axis : Angle (degrees) between LPA-RPA axis and Nasion-mid LPA-RPA point axis
        - side too long : One side of the triangle is longer than the sum of the two others
        - nasion lower : Nasion is lower than LPA/RPA
        - negative voxel : A voxel is negative
        - voxel on the edge : The voxel is too close from the edge of the volume
        - LPA-RPA diff z : Height difference between LPA and RPA
    """
    # Path of the processing directory (to change)
    directory = "250_2025-07-31"
    std_criteria = 3
    reference_distribution_for_std_cut = True
    
    # ---------------- END OF USER SETTINGS ---------------- #


    # Create a folder list to get points{...}.csv
    csv_paths = create_list(directory=directory)
    csv_paths.sort(key=lambda x: int(x.split('\\')[-1]))
    # print(csv_paths)

    dict_reg_scans = make_dict_scans(csv_paths)
    dict_props2check = find_caracteristics(dict_reg_scans)
    dict_std_criteria = dict_props2check

    # Assuming '250_CQ.json' is the JSON file
    with open('250_CQ.json', 'r') as file:
        dict_ref = json.load(file)
    dict_ref = find_caracteristics(dict_ref)
    if reference_distribution_for_std_cut:
        dict_std_criteria = dict_ref

    cut_criteria(dict_props2check, dict_reference=dict_std_criteria, std_criteria=std_criteria)
    # cut_criteria(dict_ref, dict_ref, std_criteria=2)

 
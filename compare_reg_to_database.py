import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from analyze_points import compute_distance
from analyze_points import compute_components
from automatically_get_folders import create_list
from view_results import load_landmarks_from_csv


# Assuming '250_CQ.json' is the JSON file containing the reference points
with open('250_CQ.json', 'r') as file:
    dict_scans = json.load(file)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def analyze_errors_scans(csv_dirs, remove_dim = False, too_far=10):
    """
    Compare landmark points in CSVs to reference NAS/LPA/RPA and collect errors.

    Parameters
    ----------
    csv_dirs : iterable of str
        Directories containing one CSV each (points{...}.csv) with landmark data.
    remove_dim : bool, optional
        If True, zeroes out a specified axis ("x" for NAS, "y" for LPA/RPA) when computing distances/components.
    too_far : float, optional
        Distance threshold above which a point is flagged as too far.

    Returns
    -------
    dict_err_distance : dict
        Per-landmark list of scalar distances to the reference.
    dict_err_components : dict
        Per-landmark list of component-wise differences (scaled).
    dict_too_far : dict
        Mapping from scan label (path) to a tuple (folder_number, distance) for points exceeding `too_far`.
    """
    dict_err_distance = {}
    dict_err_components = {}
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
                    components = compute_components(dict_landmarks["NAS"], pt1, res, None)
                    if remove_dim:
                        distance = compute_distance(pt1, dict_landmarks["NAS"], res, "x")
                        components = compute_components(dict_landmarks["NAS"], pt1, res, "x")
                elif i in [8,9,14,15]:
                    distance = compute_distance(pt1, dict_landmarks["LPA"], res, None)
                    components = compute_components(dict_landmarks["LPA"], pt1, res, None)
                    if remove_dim:
                        distance = compute_distance(pt1, dict_landmarks["LPA"], res, "y")
                        components = compute_components(dict_landmarks["LPA"], pt1, res, "y")
                else:
                    distance = compute_distance(pt1, dict_landmarks["RPA"], res, None)
                    components = compute_components(dict_landmarks["RPA"], pt1, res, None)
                    if remove_dim:
                        distance = compute_distance(pt1, dict_landmarks["RPA"], res, "y")
                        components = compute_components(dict_landmarks["LPA"], pt1, res, "y")
                if distance is None:
                    continue
                dict_err_distance.setdefault(name_pt, []).append(distance)
                dict_err_components.setdefault(name_pt, []).append(components)
                if distance > too_far:
                    dict_too_far[label]= (i, distance)
    return dict_err_distance, dict_err_components, dict_too_far


def show_boxplots(dict_errors):
    """
    Plot boxplots of landmark errors for Nasion, LPA, and RPA.

    Parameters
    ----------
    dict_errors : dict
        Must contain lists of errors (in mm) with keys like:
        'MRI Nasion improved (voxel)', 'MRI Nasion registered (voxel)',
        'CT LPA improved (voxel)', etc., for each landmark and stage.

    Notes
    -----
    - For each landmark, compares MRI vs CT and improved vs registered versions.
    - Prints median error for MRI and CT before showing the boxplot.
    """
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
        print("Median MRI", np.median(landmarks[pt_name][0]))
        print("Median CT", np.median(landmarks[pt_name][2]))

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


def show_error_vectors(big_output_directory, reg_with="MRI",landmarks_type="reg"):
    """
    Compute and plot registration error vectors for NAS, LPA, and RPA across cases.

    Parameters
    ----------
    big_output_directory : str
        Root folder containing per-case subdirectories with landmark CSVs (points{...}.csv) .
    reg_with : str, default="MRI"
        Modality key to use from the loaded landmarks ("MRI" or "CT").
    landmarks_type : str, default="reg"
        Subtype of landmarks to use ("reg" or "imp").

    Notes
    -----
    - For each case, compares the registered landmark to a reference from `dict_scans`,
      computes per-axis displacement vectors, and accumulates them.
    - Plots individual error vectors per landmark and the aggregate vector sum.
    """
    # lit of dictionnaries for nas, lpa, rpa
    all_dict = [{k: [] for k in ("x_origins", "y_origins", "z_origins", "u_comps", "v_comps", "w_comps")} for _ in range(3)]
    colors = ["green", "blue", "red"]

    folder_path_list = create_list(big_output_directory)

    for folder_path in tqdm(folder_path_list):
        file_number = os.path.basename(folder_path)
        csv_path = os.path.join(folder_path, "points"+file_number+".csv")
        dict_csv = load_landmarks_from_csv(csv_path=csv_path)
        scan_name = dict_csv["scan name"]
        res = np.array(dict_scans[scan_name]["res"])
        nas, lpa, rpa = [pt*res for pt in np.array(dict_csv[reg_with][landmarks_type])]
        my_nas = np.array(dict_scans[scan_name]["NAS"])*res
        my_lpa = np.array(dict_scans[scan_name]["LPA"])*res
        my_rpa = np.array(dict_scans[scan_name]["RPA"])*res

        points = [nas, lpa, rpa]
        my_points = [my_nas, my_lpa, my_rpa]
        
        # Compute vectors
        for i in range(3):
            x, y, z = points[i]
            u, v, w = np.array(my_points[i]) - np.array(points[i]) 
            all_dict[i]["x_origins"].append(x)
            all_dict[i]["y_origins"].append(y)
            all_dict[i]["z_origins"].append(z)
            all_dict[i]["u_comps"].append(u)
            all_dict[i]["v_comps"].append(v)
            all_dict[i]["w_comps"].append(w)

    w_tot = []
    u_tot = []

    # Show vectors (difference between my_point and point found with registration) for a landmark
    for i in range(3):
        x_origins, y_origins, z_origins = all_dict[i]["x_origins"], all_dict[i]["y_origins"], all_dict[i]["z_origins"]
        u_comps, v_comps, w_comps = all_dict[i]["u_comps"], all_dict[i]["v_comps"], all_dict[i]["w_comps"] 
        plt.quiver(z_origins, x_origins, w_comps, u_comps, angles='xy', scale_units='xy', scale=1, color=colors[i])
        plt.xlim(0, 300)
        plt.ylim(0, 300)
        plt.xlabel("Z-axis (mm)")
        plt.ylabel("X-axis (mm)")
        plt.grid(True)
        plt.show()

        w_tot.append(np.sum(w_comps))
        u_tot.append(np.sum(u_comps))

    # Show the resulting vector for each landmark
    plt.quiver([0,0,0], [0,0,0], w_tot, u_tot, angles='xy', scale_units='xy', scale=1, color=colors)
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.xlabel("Z-axis (mm)")
    plt.ylabel("X-axis (mm)")
    plt.grid(True)
    plt.show()


def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + \
            (y - mu_y)**2 / (2 * sigma_y**2)))


def show_error_distribution(dict_err_components):
    """
    Visualize the 2D error distribution (x vs z) for landmark component errors.

    Parameters
    ----------
    dict_err_components : dict
        Mapping from landmark names to lists of (dx, dy, dz) error tuples.

    Notes
    -----
    - Skips nasion.
    - Fits a 2D Gaussian to the x/z errors and plots its contour with the raw points.
    - Displays mean shifts in the plot.
    """
    x = []
    z = []
    for i, key in enumerate(dict_err_components):
        if i in [0,1,6,7] : # Skip nasion
            continue
        x_list = [pt[0] for pt in dict_err_components[key]]
        y_list = [pt[1] for pt in dict_err_components[key]]
        z_list = [pt[2] for pt in dict_err_components[key]]

        x = np.linspace(-30, 30, 500) # Adjust range and number of points as needed
        y = np.linspace(-30, 30, 500)
        X, Y = np.meshgrid(x, y)

        mu_x, mu_y = np.mean(x_list), np.mean(z_list)  # Mean values
        sigma_x, sigma_y = np.std(x_list), np.std(z_list)  # Standard deviations
        Z = gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y)
        plt.figure()#figsize=(8, 6)
        plt.contourf(X, Y, Z, cmap='viridis', levels=100) # levels controls number of contour lines
        plt.scatter(x_list,z_list, color="red", linewidths=0.5)
        plt.text(-18, -18, f'voxel shift\n x : {mu_x}\n z : {mu_y}', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.colorbar(label='Probability Density')
        # plt.text(0, 0, f'voxel shift\n x : {mu_x}\n z : {mu_y}', fontsize=12, color='black')
        plt.xlabel('error for X-axis (voxel)')
        plt.ylabel('error for Z-axis (voxel)')
        plt.title(f'2D Gaussian error for {key}')
        plt.grid(True)
        plt.show()


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------ USER SETTINGS ------------------ #
    # Choose working folder and file number
    big_output_directory = "250_2025-07-31" # Folder with NIfTI and CSV files

    # Error in mm considered as not acceptable compared to the reference
    # path : (folder_number, distance)
    too_far = 10

    # For the vector function
    see_error_vectors = True
    # Choose which landmarks to display (can be CT or MRI, reg or imp)
    reg_with = "MRI" # "CT" or "MRI"
    landmarks_type = "reg" # "reg" or "imp"

    # For the gaussian error distribution 
    see_error_distribution = True

    # For the boxplots
    see_boxplots = True
    # ---------------- END OF USER SETTINGS ---------------- #

    if see_error_vectors:
        show_error_vectors(big_output_directory, reg_with, landmarks_type)

    # Create a folder list to get points{...}.csv
    csv_dirs = create_list(directory=big_output_directory)
    csv_dirs.sort(key=lambda x: int(x.split('\\')[-1]))

    dict_err_distance, dict_err_components, dict_too_far = analyze_errors_scans(csv_dirs=csv_dirs, remove_dim=True, too_far=too_far)
    print("Scans considered as too far", dict_too_far)

    if see_error_distribution:
        show_error_distribution(dict_err_components)

    if see_boxplots:
        show_boxplots(dict_errors=dict_err_distance)


import os
import pandas as pd
import numpy as np

from automatically_get_folders import create_list


def clean_value(val):
    """
    Clean and normalize values extracted from a CSV file.

    This function transforms NaN or '-' values into None. Otherwise, it returns the original value.

    Parameters
    ----------
    val : any
        A value from a DataFrame cell.

    Returns
    -------
    val : any or None
        The cleaned value: None if the input was NaN or '-', otherwise unchanged.
    """
    if pd.isna(val):
        return None
    if isinstance(val, str) and val.strip() == '-':
        return None
    return val


def create_summary(csv_paths, output_dir):
    """
    Generate a summary CSV file compiling landmarks, resolutions, and metadata from multiple folders.

    This function looks for `.csv` files in given folders. These `.csv` contain anatomical landmarks
    (nasion, LPA, RPA) in voxel space, image resolutions, and processing times. The extracted data are aggregated 
    into a single summary CSV.

    Parameters
    ----------
    csv_paths : list of str
        List of paths to folders expected to contain `.csv` files.
    output_dir : str
        Directory where the final summary file `summary.csv` will be saved.

    Notes
    -----
    - If a `.csv` file is not found in a folder, None is used for all fields.
    - Columns include: resolution, processing time, MRI landmarks, CT landmarks, and DICOM path.
    - Landmarks are reported as x, y, z for improved and registered positions.
    """
    list_of_dict = []
    list_segm_processing_time, list_ct_processing_time, list_mri_processing_time = [], [], []

    # Columns of interest
    points = ['MRI NAS imp', 'MRI NAS reg',
        'MRI LPA imp', 'MRI LPA reg',
        'MRI RPA imp', 'MRI RPA reg',
        'CT NAS imp', 'CT NAS reg',
        'CT LPA imp', 'CT LPA reg',
        'CT RPA imp', 'CT RPA reg' ]
        

    for csv_path in csv_paths:
        csv_file = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        big_dict_csv = {}

        big_dict_csv["path"] = csv_path

        if not csv_file:
            # If no .csv, puts None everywhere
            big_dict_csv["processing time"] = None
            big_dict_csv["dcm path"] = None
            big_dict_csv["res_x"] = None
            big_dict_csv["res_y"] = None
            big_dict_csv["res_z"] = None
            for point in points:
                for axis in ['x', 'y', 'z']:
                    big_dict_csv[f"{point} {axis}"] = None
            list_of_dict.append(big_dict_csv)
            continue

        # Else, loads the .csv file
        csv_file_path = os.path.join(csv_path, csv_file[0])
        df = pd.read_csv(csv_file_path, sep=",", header=None, on_bad_lines='skip')

        # dcm path (row 1)
        print(df.iloc[1,0])
        big_dict_csv["dcm path"] = df.iloc[1, 0]

        # Processing time (row 18-19-20)
        segm_processing_time = clean_value(df.iloc[20, 1]) if len(df) > 20 and len(df.columns) > 2 else None
        big_dict_csv["segmentation processing time"] = segm_processing_time
        ct_processing_time = clean_value(df.iloc[19, 1]) if len(df) > 19 and len(df.columns) > 2 else None
        big_dict_csv["CT processing time"] = ct_processing_time
        mri_processing_time = clean_value(df.iloc[18, 1]) if len(df) > 18 and len(df.columns) > 2 else None
        big_dict_csv["MRI processing time"] = mri_processing_time
        if segm_processing_time is not None and mri_processing_time is not None and ct_processing_time is not None:
            list_segm_processing_time.append(float(segm_processing_time))
            list_ct_processing_time.append(float(ct_processing_time))
            list_mri_processing_time.append(float(mri_processing_time))

        # Resolution (row 3)
        res_x, res_y, res_z = (None, None, None)
        dim_x, dim_y, dim_z = (None, None, None)
        if len(df) > 4:
            res_x, res_y, res_z = map(clean_value, df.iloc[4, 1:4])
            dim_x, dim_y, dim_z = map(clean_value, df.iloc[2, 1:4])
        big_dict_csv["res_x"] = res_x
        big_dict_csv["res_y"] = res_y
        big_dict_csv["res_z"] = res_z
        big_dict_csv["dim_x"] = dim_x
        big_dict_csv["dim_y"] = dim_y
        big_dict_csv["dim_z"] = dim_z

        # MRI
        start_block = 6
        mri_block = df.iloc[start_block:18]
        if not mri_block.empty:
            for offset, point in enumerate(points):
                line_index=start_block+offset
                values = df.iloc[line_index, 1:4].map(clean_value)
                big_dict_csv[f"{point} x"] = values.iloc[0]
                big_dict_csv[f"{point} y"] = values.iloc[1]
                big_dict_csv[f"{point} z"] = values.iloc[2]
        else:
            for point in points:
                big_dict_csv[f"{point} x"] = None
                big_dict_csv[f"{point} y"] = None
                big_dict_csv[f"{point} z"] = None

        list_of_dict.append(big_dict_csv)

    # Final dataframe
    df_final = pd.DataFrame(list_of_dict)
    df_final.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    return np.array(list_segm_processing_time), np.array(list_ct_processing_time), np.array(list_mri_processing_time)
    

# ---------- USER SECTION: Only modify parameters below this line ----------
if __name__ == "__main__":
    # Path of the processing directory (to change)
    directory = "250_2025-07-31" 
    # Create a folder list to get points{...}.csv
    csv_paths = create_list(directory=directory)
    csv_paths.sort(key=lambda x: int(x.split('\\')[-1]))
    print(csv_paths)

    # Create summary.csv
    segm_processing_time, ct_processing_time, mri_processing_time = create_summary(csv_paths=csv_paths, output_dir=directory)
    print(segm_processing_time)

    print("Average segmentation processing time (s) :", np.mean(segm_processing_time),"±",2*np.std(segm_processing_time))
    print("Average identification processing time with MRI (s) :", np.mean(mri_processing_time-segm_processing_time),"±",2*np.std(mri_processing_time-segm_processing_time))
    print("Average identification processing time with CT (s) :", np.mean(mri_processing_time-ct_processing_time),"±",2*np.std(mri_processing_time-ct_processing_time))
    print("Average time - total (s) :", np.mean(mri_processing_time),"±",2*np.std(mri_processing_time))
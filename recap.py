

import os
import pandas as pd
from automatically_get_dicom_folders import create_list

def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, str) and val.strip() == '-':
        return None
    return val

# Génère ta liste des dossiers
csv_paths = create_list(directory="50_2025-07-17")
csv_paths.sort(key=lambda x: int(x.split('\\')[-1]))

list_of_dict = []

# Colonnes cibles
points = ['Nasion improved (voxel)', 'Nasion registered (voxel)',
          'LPA improved (voxel)', 'LPA registered (voxel)',
          'RPA improved (voxel)', 'RPA registered (voxel)']
blocks = ['mri', 'ct']

for csv_path in csv_paths:
    csv_file = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
    big_dict_csv = {}

    # filename = os.path.basename(csv_path)
    # big_dict_csv["filename"] = filename
    big_dict_csv["path"] = csv_path

    if not csv_file:
        # Si aucun .csv, mets None partout
        big_dict_csv["processing time"] = None
        big_dict_csv["dcm path"] = None
        big_dict_csv["res_x"] = None
        big_dict_csv["res_y"] = None
        big_dict_csv["res_z"] = None
        for block in blocks:
            for point in points:
                for coord in ['x', 'y', 'z']:
                    big_dict_csv[f"{block} {point} {coord}"] = None
        list_of_dict.append(big_dict_csv)
        continue

    # Sinon, charge le fichier CSV
    csv_file_path = os.path.join(csv_path, csv_file[0])
    df = pd.read_csv(csv_file_path, sep=",", header=None)

    # dcm path (ligne 1)
    print(df.iloc[1,0])
    big_dict_csv["dcm path"] = df.iloc[1, 0]

    # Processing time (ligne 23)
    processing_time = clean_value(df.iloc[23, 1]) if len(df) > 23 and len(df.columns) > 2 else None
    big_dict_csv["processing time"] = processing_time

    # Résolution (ligne 3)
    res_x, res_y, res_z = (None, None, None)
    if len(df) > 3:
        res_x, res_y, res_z = map(clean_value, df.iloc[4, 1:4])
    big_dict_csv["res_x"] = res_x
    big_dict_csv["res_y"] = res_y
    big_dict_csv["res_z"] = res_z

    # MRI
    mri_block_idx = df[df[0].str.contains('normalized MRI', na=False)].index
    if not mri_block_idx.empty:
        idx = mri_block_idx[0]
        for offset, point in enumerate(points, start=1):
            line_idx = idx + offset
            if len(df) > line_idx:
                values = df.iloc[line_idx, 1:4].map(clean_value)
                big_dict_csv[f"mri {point} x"] = values.iloc[0]
                big_dict_csv[f"mri {point} y"] = values.iloc[1]
                big_dict_csv[f"mri {point} z"] = values.iloc[2]
            else:
                big_dict_csv[f"mri {point} x"] = None
                big_dict_csv[f"mri {point} y"] = None
                big_dict_csv[f"mri {point} z"] = None
    else:
        for point in points:
            big_dict_csv[f"mri {point} x"] = None
            big_dict_csv[f"mri {point} y"] = None
            big_dict_csv[f"mri {point} z"] = None

    # CT
    ct_block_idx = df[df[0].str.contains('non-normalized CT scan', na=False)].index
    if not ct_block_idx.empty:
        idx = ct_block_idx[0]
        for offset, point in enumerate(points, start=1):
            line_idx = idx + offset
            if len(df) > line_idx:
                values = df.iloc[line_idx, 1:4].map(clean_value)
                big_dict_csv[f"ct {point} x"] = values.iloc[0]
                big_dict_csv[f"ct {point} y"] = values.iloc[1]
                big_dict_csv[f"ct {point} z"] = values.iloc[2]
            else:
                big_dict_csv[f"ct {point} x"] = None
                big_dict_csv[f"ct {point} y"] = None
                big_dict_csv[f"ct {point} z"] = None
    else:
        for point in points:
            big_dict_csv[f"ct {point} x"] = None
            big_dict_csv[f"ct {point} y"] = None
            big_dict_csv[f"ct {point} z"] = None

    list_of_dict.append(big_dict_csv)

# Crée le DataFrame final
df_final = pd.DataFrame(list_of_dict)
df_final.to_csv("50_2025-07-17/summary.csv", index=False)

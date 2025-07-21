import os
import re


def find_dcm_folders(directory):
    """
    For each folder in the given directory, this function collects all paths to subfolders 
    that do not contain further subfolders (i.e., leaf folders containing DICOMs).

    Parameters
    ----------
    directory : str
        Root directory to search for DICOM folders.

    Returns
    -------
    list of lists of str
        A list containing lists of DICOM folder paths for each subdirectory.
    """
    # List all the folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    all_dcm_folders =[]    
    for folder in folders:
        dcm_folders = []
        for root, dirs, files in os.walk(os.path.join(directory,folder)):
            # Keep only folders without subfolders
            if len(dirs) == 0:
                dcm_folders.append(root)
        all_dcm_folders.append(dcm_folders)
    return all_dcm_folders


def conservation_criteria(dcm_folders, green_flags, red_flags):
    """
    Select the most appropriate DICOM folder based on name criteria.

    - If there is only one folder, returns it directly.
    - If multiple folders:
        - Prefers folders containing 'green_flags' keywords, unless they also contain 'red_flags'.
        - Otherwise selects the folder with the smallest numerical value (often resolution) in its name.
        - If none contain numbers, returns the first folder not matching red flags.

    Parameters
    ----------
    dcm_folders : list of str
        List of DICOM folder paths to evaluate.
    green_flags : list of str
        List of keywords (in lowercase) considered as preferred (green flags).
    red_flags : list of str
        List of keywords (in lowercase) considered to exclude (red flags).

    Returns
    -------
    str or None
        The selected DICOM folder path or None if none is suitable.
    """        
    # If there is only one file containing dicoms, directly takes it
    if len(dcm_folders) == 1:
        return dcm_folders[0]

    elif len(dcm_folders) > 1:
        dict_resolution = {}
        list_nothing_special = []
        for dcm_folder_path in dcm_folders:
            # If a green flag is found in the name of the path, directly takes it
            if any(green_flag in dcm_folder_path.lower() for green_flag in green_flags):
                if not any(red_flag in dcm_folder_path.lower() for red_flag in red_flags):
                    return dcm_folder_path     
            else:
                # Extract numerical values for sorting by resolution
                basename = os.path.basename(dcm_folder_path)
                number_str = re.findall(r"[-+]?(?:\d*\.*\d+)", basename)
                number = [abs(float(i)) for i in number_str]
                if not any(red_flag in basename.lower() for red_flag in red_flags):
                    if number: 
                        dict_resolution[number[0]] = dcm_folder_path
                    else:
                        list_nothing_special.append(dcm_folder_path)
        if dict_resolution:
            return dict_resolution[min(dict_resolution.keys())]
        elif len(list_nothing_special) > 0:
            return list_nothing_special[0]

    return None


def create_list(directory="dicom_dataset", green_flags = ["thin"], red_flags=["bone", "bones", "std", "oral", "sec"]):
    """
    Generate a list of DICOM folders selected according to specific criteria.

    For each subdirectory in the given root directory:
        - Finds all DICOM folders.
        - Selects the best one according to conservation criteria.

    Parameters
    ----------
    directory : str, optional
        Root directory to search for DICOM folders. Defaults to "dicom_dataset".
    green_flags : list of str, optional
        Keywords (in lowercase) indicating preferred folders. Defaults to ["thin"].
    red_flags : list of str, optional
        Keywords (in lowercase) indicating folders to exclude. Defaults to ["bone", "bones", "std", "oral", "sec"].

    Returns
    -------
    list of str
        List of selected DICOM folder paths.
    """
    final_folders = []
    all_dcm_folders = find_dcm_folders(directory=directory)
    for dcm_folders in all_dcm_folders:
        final_folders.append(conservation_criteria(dcm_folders=dcm_folders, green_flags=green_flags, red_flags=red_flags))
    return final_folders

            

if __name__ == "__main__":
    dicoms_list = create_list(directory="50_CQ")
    print(dicoms_list)


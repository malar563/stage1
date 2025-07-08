import os
import re


def create_dicom_list(directory="jspakoi", green_words = ["THIN", "thin"]):
    # print("function called")
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    final_folders = []
    dict_resolution = {}

    for folder in folders:
        dcm_folders = []
        for root, dirs, files in os.walk(os.path.join(directory,folder)):
            # print("Root:", root)
            # print("Directories:", dirs)
            # print("Files:", files)
            # print("-"*30)

            # Only takes folders with no subfolders
            if len(dirs) == 0:
                dcm_folders.append(root)

        # If there is only one file containing dicoms, directly takes it
        if len(dcm_folders) == 1:
            final_folders.append(dcm_folders[0])
        
        elif len(dcm_folders) > 1:
            for dcm_folder_path in dcm_folders:
                
                # If a green word is found in the name of the path, directly takes it
                if any(green_word in dcm_folder_path for green_word in green_words):
                    final_folders.append(dcm_folder_path)

                # Takes the file with the smallest number (resolution) mentionned
                basename = os.path.basename(dcm_folder_path)
                number_str = re.findall(r"[-+]?(?:\d*\.*\d+)", basename)
                number = [float(i) for i in number_str]
                if len(number) != 0:
                    dict_resolution[number[0]] = dcm_folder_path
            try:
                final_folders.append(dict_resolution[min(dict_resolution.keys())])
            except:
                pass # Can't put final_folders.append(None) here because it will appear if there is a green word too
        else:
            final_folders.append(None)

    return final_folders
                

if __name__ == "__main__":
    create_dicom_list
import os
import re



def find_dcm_folders(directory="50_CQ"):
    # print("function called")
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    all_dcm_folders =[]
        
    i=0
    for folder in folders:
        dcm_folders = []
        print(i)
        i+=1
        for root, dirs, files in os.walk(os.path.join(directory,folder)):
            # print("Root:", root)
            # print("Directories:", dirs)
            # print("Files:", files)
            # print("-"*30)

            # Only takes folders with no subfolders
            if len(dirs) == 0:
                dcm_folders.append(root)
        all_dcm_folders.append(dcm_folders)
    return all_dcm_folders


def conservation_criteria(dcm_folders, green_flags = ["thin"], red_flags=["bone", "bones", "std", "oral", "sec"]):
        
    # If there is only one file containing dicoms, directly takes it
    if len(dcm_folders) == 1:
        return dcm_folders
        
    elif len(dcm_folders) > 1:
        print("est-ce que dcm folders a mm no :", dcm_folders)
        dict_resolution = {}
        list_nothing_special = []
        for dcm_folder_path in dcm_folders:
            # If a green word is found in the name of the path, directly takes it
            if any(green_flag in dcm_folder_path.lower() for green_flag in green_flags):
                print("OUIIII")
                if not any(red_flag in dcm_folder_path.lower() for red_flag in red_flags):
                    return dcm_folder_path # <----- Has to
                    
            else:

                # Takes the file with the smallest number (resolution) mentionned
                basename = os.path.basename(dcm_folder_path)
                number_str = re.findall(r"[-+]?(?:\d*\.*\d+)", basename, re.IGNORECASE)
                number = [abs(float(i)) for i in number_str]
                if len(number) != 0:
                    if not any(red_flag in basename.lower() for red_flag in red_flags):
                        dict_resolution[number[0]] = dcm_folder_path
                else:
                    if not any(red_flag in basename.lower() for red_flag in red_flags):
                        list_nothing_special.append(dcm_folder_path)
                        print("list nothing special",list_nothing_special)

        try:
            if len(dict_resolution) > 0:
                print(dict_resolution)
                return dict_resolution[min(dict_resolution.keys())]
            elif len(list_nothing_special) > 0:
                return list_nothing_special[0]
        except:
            pass # Can't put final_folders.append(None) here because it will appear if there is a green word too
    return None


def create_list(directory="50_CQ"):
    all_dcm_folders = find_dcm_folders(directory=directory)
    print(all_dcm_folders)
    final_folders = []
    for i, dcm_folders in enumerate(all_dcm_folders):
        final_folders.append(conservation_criteria(dcm_folders=dcm_folders))
    return final_folders

dcm_list = create_list()
print(dcm_list, len(dcm_list))
                





# if __name__ == "__main__":
#     create_dicom_list
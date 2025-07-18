import os
import re


# def create_dicom_list(directory="jspakoi", green_words = ["THIN", "thin", "Thin"]):
#     # print("function called")
#     folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
#     final_folders = []
    

#     i=0
#     for folder in folders:
#         dcm_folders = []
#         print(i)
#         i+=1
#         for root, dirs, files in os.walk(os.path.join(directory,folder)):
#             # print("Root:", root)
#             # print("Directories:", dirs)
#             # print("Files:", files)
#             # print("-"*30)

#             # Only takes folders with no subfolders
#             if len(dirs) == 0:
#                 dcm_folders.append(root)

#         # If there is only one file containing dicoms, directly takes it
#         if len(dcm_folders) == 1:
#             final_folders.append(dcm_folders[0])
        
#         elif len(dcm_folders) > 1:
#             print("est-ce que dcm folders a mm no :", dcm_folders)
#             dict_resolution = {}
#             for dcm_folder_path in dcm_folders:
                
#                 # If a green word is found in the name of the path, directly takes it
#                 if any(green_word in dcm_folder_path for green_word in green_words):
#                     print("OUIIII")
#                     final_folders.append(dcm_folder_path)

#                 else:
#                     try:
#                         # Takes the file with the smallest number (resolution) mentionned
#                         basename = os.path.basename(dcm_folder_path)
#                         number_str = re.findall(r"[-+]?(?:\d*\.*\d+)", basename)
#                         number = [abs(float(i)) for i in number_str]
#                         if len(number) != 0:
#                             dict_resolution[number[0]] = dcm_folder_path
#                     except:
#                         final_folders.append(None) # If no number is found
#             try:
#                 print(dict_resolution)
#                 final_folders.append(dict_resolution[min(dict_resolution.keys())])
#             except:
#                 pass # Can't put final_folders.append(None) here because it will appear if there is a green word too
#         else:
#             final_folders.append(None)

#     return final_folders



def find_dcm_folders(directory="50_CQ"):
    # print("function called")
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    all_dcm_folders =[]
        
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
        all_dcm_folders.append(dcm_folders)
    return all_dcm_folders


def conservation_criteria(dcm_folders, green_flags, red_flags):
        
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
                    return dcm_folder_path # <----- Has to
                    
            else:
                # Takes the file with the smallest number (resolution) mentionned
                basename = os.path.basename(dcm_folder_path)
                number_str = re.findall(r"[-+]?(?:\d*\.*\d+)", basename)
                number = [abs(float(i)) for i in number_str]
                if not any(red_flag in basename.lower() for red_flag in red_flags):
                    if len(number) != 0: 
                        dict_resolution[number[0]] = dcm_folder_path
                    else:
                        list_nothing_special.append(dcm_folder_path)

        if len(dict_resolution) > 0:
            return dict_resolution[min(dict_resolution.keys())]
        elif len(list_nothing_special) > 0:
            return list_nothing_special[0]

    return None


def create_list(directory="50_CQ", green_flags = ["thin"], red_flags=["bone", "bones", "std", "oral", "sec"]):
    final_folders = []
    all_dcm_folders = find_dcm_folders(directory=directory)
    for i, dcm_folders in enumerate(all_dcm_folders):
        final_folders.append(conservation_criteria(dcm_folders=dcm_folders, green_flags=green_flags, red_flags=red_flags))
    # final_folders.sort(key=lambda x: int(x.split('\\')[1]))
    return final_folders

dcm_list = create_list()
print(dcm_list, len(dcm_list))
                





                

if __name__ == "__main__":
    dicoms_list = create_list(directory="50_CQ")
    print(dicoms_list)

# dicom_list = create_dicom_list(directory="50_CQ")
# print(dicom_list, len(dicom_list))
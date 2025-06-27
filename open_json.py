import json
import pydicom
import os
import dicom2nifti

# Open and read the JSON file
# with open('MIDRC_case_manifest.json', 'r') as file:
with open('MIDRC_Cases_table.json', 'r') as file:
    data = json.load(file)

# Print the data
# print(data)

ds = pydicom.dcmread("online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463/2.16.840.1.114274.1818.461189614183831996910780981623526981287.dcm")
ds = pydicom.dcmread("online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463/2.16.840.1.114274.1818.461276342881975299517310158650044366255.dcm")
ds = pydicom.dcmread("online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463/2.16.840.1.114274.1818.461404773312228968316497215340622248637.dcm")
ds = pydicom.dcmread("online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463/2.16.840.1.114274.1818.461488091919503678218015278656383507871.dcm")

# print(ds)


file_list = os.listdir("online_patient/test")
print(file_list)
for file in file_list:
    ds = pydicom.dcmread("online_patient/test/"+file)
    os.rename("online_patient/test/"+file, f"online_patient/test/{ds.InstanceNumber}.dcm")
    print(ds.InstanceNumber)

dicom2nifti.convert_directory("online_patient/test", "online/0", compression=True)
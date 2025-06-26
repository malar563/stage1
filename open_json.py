import json
import pydicom

# Open and read the JSON file
# with open('MIDRC_case_manifest.json', 'r') as file:
with open('MIDRC_Cases_table.json', 'r') as file:
    data = json.load(file)

# Print the data
# print(data)

ds = pydicom.dcmread("online_patient/2.16.840.1.114274.1818.46711723837672246304206241465856141463/2.16.840.1.114274.1818.495584876609101762288809117582729631.dcm")
print(ds)
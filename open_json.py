import json

# Open and read the JSON file
# with open('MIDRC_case_manifest.json', 'r') as file:
with open('MIDRC_Cases_table.json', 'r') as file:
    data = json.load(file)

# Print the data
print(data)
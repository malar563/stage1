import zipfile 
import os 


output_directory = r"D:\MaryliseLarouche\CQ500\50_CQ" 
zip_directory = r"D:\MaryliseLarouche\CQ500\50_CQ" 

 
zip_files = [f for f in os.listdir(zip_directory) if f.endswith(".zip")] 
print(zip_files) 

 
for i, zip_file in enumerate(zip_files): 
    zip_file = os.path.join(zip_directory, zip_file) 
    with zipfile.ZipFile(zip_file, 'r') as zip_ref: 
        zip_ref.extractall(output_directory) 
    os.remove(zip_file)
    print(i)


 
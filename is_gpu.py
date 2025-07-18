import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import pandas as pd

data = [{'Name': 'Alice', 'Age': 25, 'City': 'New York'},
        {'Name': 'Bob', 'Age': 30, 'City': 'London'},
        {'Name': 'Charlie', 'Age': 22, 'City': 'Paris'}]
df = pd.DataFrame(data)
print(df)

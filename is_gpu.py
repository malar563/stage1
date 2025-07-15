import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    for i in range(device_count):
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")
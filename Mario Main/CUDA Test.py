import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)         # Should return 12.1
print(torch.__version__)   
print(torch.cuda.get_device_name(0))
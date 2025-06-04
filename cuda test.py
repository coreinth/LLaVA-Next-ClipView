import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")  # Should match your NVCC (12.8)
print(f"GPU: {torch.cuda.get_device_name(0)}")

import torch

device=torch.device("cuda")

x = torch.randn(5, 3)
y = torch.randn(5, 3)

x = x.to(device)
y = y.to(device)
z = x + y
print("GPU上での計算結果:")
print(z)

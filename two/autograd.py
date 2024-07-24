import torch
import math


def sin(x):
    return 2 * torch.dot(x, x)


x = torch.arange(4.0)
x.requires_grad_(True)

y = sin(x)
y.backward()
print(x.grad)
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
# from d2l import torch as

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)

mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True
)

# print(len(mnist_train), len(mnist_test))



batch_size=256
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
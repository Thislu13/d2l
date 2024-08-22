import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data


class Accumulator:
    """For accumulating sums over 'n' variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):

    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
        metric = Accumulator(2)
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0]/metric[1]


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_block = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_block.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_block,
        nn.Flatten(),
        nn.Linear(out_channels*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reduce_sum =  lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *arge, **kwargs: x.numel(*arge, **kwargs)


conv_arch = ((1,64), (1, 128), (2, 256), (2, 512), (2, 512))


net = vgg(conv_arch)

# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'shape:\t', X.shape)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

batch_size = 128
num_epochs = 10
device = 'cuda:0'
lr = 0.05

trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(244))
trans = transforms.Compose(trans)


mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

net.apply(init_weights)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    net.train()
    for i, (X, y) in enumerate(train_iter):
        metric = Accumulator(3)
        optimizer.zero_grad() 
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])

        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

    if epoch >= (num_epochs - 10):
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch:{epoch}, loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    else:
        print(f'epoch:{epoch}, loss:{train_l}, accuracy:{train_acc}')

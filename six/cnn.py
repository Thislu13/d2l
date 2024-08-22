import torch
from torch import nn




def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w] * K).sum()

    return Y

def corr2d_multi_in(x, k):
    return sum(corr2d(x, k) for x, k in zip(x, k))


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)





class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super.__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                  [[1.0, 2.0], [3.0, 4.0]]])
K = torch.stack((K, K+1, K+2), 0)

print(K.shape)

print(corr2d_multi_in_out(X, K))



# X = torch.ones((6, 8))
# X[:, 2:6] = 0
# print(X)
# K = torch.tensor([[1.0, -1.0]])
# Y = corr2d(X, K)
# print(Y)
# print(corr2d(X, K))

# conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
#
# X = X.reshape((1, 1, 6, 8))
# Y = Y.reshape((1, 1, 6, 7))
#
# for i in range(50):
#     Y_hat = conv2d(X)
#     conv2d.zero_grad()
#     l = (Y_hat - Y)**2
#
#     l.sum().backward()
#     conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
#     if (i + 1) % 2 == 0:
#         print(f'batch {i+1}, loss {l.sum():.3f}')
#         print(conv2d.weight.data)


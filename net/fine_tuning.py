from d2l import torch as d2l
import os
import torch
import torchvision
from torch import nn

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

data_path = '/media/00.Data/00.User/03.wqs/01.Code/d2l/data/hotdog'
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform_train)
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform_test)


finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):

    train_iter = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]

        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate*10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)

    # train_imgs.to(device)
    # test_iter.to(device)
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, train_acc_num, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            trainer.zero_grad()

            y_hat = net(X)
            l = loss(y_hat, y).sum()
            l.backward()

            trainer.step()

            train_loss_sum += l.item()
            train_acc_num += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]

        train_loss = train_loss_sum / n
        train_acc = train_acc_num / n


        net.eval()
        test_acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X,y in test_iter:
                X = X.to(device)
                y = y.to(device)

                y_hat = net(X)
                test_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
                n += y.shape[0]

        test_acc = test_acc_sum / n

        print(f"Epoch {epoch + 1}:  Loss:{train_loss} acc:{train_acc} test_acc:{test_acc}")




train_fine_tuning(finetune_net, 5e-5, 128, 100, True)
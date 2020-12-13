import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import json
from csv_to_json import train_mean, train_std, test_mean, test_std
from myresnet import ResModel
from torchresnet import torchResnet18

device = torch.device("cuda")
batch_size = 64

def pre_process(path, train):
    with open(path, "r") as fp:
        data_as_dict = json.load(fp)

    x = torch.tensor(data_as_dict['pixels']).view(-1,1,48,48).float()

    if train==False:
        return x
    else:
        y = torch.tensor(data_as_dict['emotion']).long()
        return x, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        '''
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        '''

        self.fully_connect = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.BatchNorm1d(4096),
            nn.RReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.BatchNorm1d(256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 7)
        )


    def forward(self, x):
        # 1 * 48 * 48 -> 32 * 24 * 24
        x = self.conv1(x)
        # 32 * 24 * 24 -> 64 * 12 * 12
        x = self.conv2(x)
        # 64 * 12 * 12 -> 256 * 6 * 6
        x = self.conv3(x)

        #x = self.conv4(x)

        x = x.view(-1, 256*6*6)
        x = self.fully_connect(x)
        return x

def myResNet(channel):
    backbone = models.resnet50(pretrained=False)
    backbone.conv1 = nn.Conv2d(1, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)

    return backbone

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.
    correct = 0.
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = F.cross_entropy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += F.cross_entropy(pred, target, reduction="sum").item()
            prediction = pred.argmax(dim=1)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item()))

    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset) * 100
    print("Train Epoch: {}, Train_loss: {}, Train_accuracy: {}".format(epoch, train_loss, train_acc))

def validation(model, device, vali_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(vali_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(vali_loader.dataset)
    acc = correct / len(vali_loader.dataset)*100
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))



if __name__ == "__main__":

    x,y = pre_process("train.json", True)
    train_x = x[2800:]
    train_y = y[2800:]
    vali_x = x[:2800]
    vali_y = y[:2800]
    #print(train_x.size(), train_y.size())
    #print(vali_x.size(), vali_y.size())

    #test_x = pre_process("test.json", False)
    #print(train_y.size())

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    vali_dataset = torch.utils.data.TensorDataset(vali_x, vali_y)
    #print(train_dataset[0][0].size(), train_dataset[0][1].size())
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=1, pin_memory=True)
    vali_dataloader = torch.utils.data.DataLoader(dataset=vali_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=1, pin_memory=True)

    #print(train_dataloader)
    #print(vali_dataloader)

    lr = 0.01
    momentum = 0.5
    decay = 1e-6
    #model = Net().to(device)
    #model = ResModel(7).to(device)
    model = torchResnet18(7, 0.2)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    num_epochs = 60
    for epoch in range(num_epochs):
        train(model, device, train_dataloader, optimizer1, epoch)
        validation(model, device, vali_dataloader)

    torch.save(model,"my_model2.pth")

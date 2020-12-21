import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
from train import Net
from myresnet import ResModel
from train import device
from train import batch_size
from train import pre_process
from train import emotion



def type_acc(loader, type):
    model.eval()
    correct = 0.
    num_type = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            for i in range(len(target)):
                if int(target[i].item())==type:
                    num_type = num_type + 1
                    correct += pred[i].eq(target[i].view_as(pred[i])).sum().item()

    acc = correct / num_type * 100

    return num_type, acc


x, y = pre_process("train.json", True)
train_x = x[2800:]
train_y = y[2800:]
vali_x = x[:2800]
vali_y = y[:2800]
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True)
vali_dataset = torch.utils.data.TensorDataset(vali_x, vali_y)
vali_dataloader = torch.utils.data.DataLoader(dataset=vali_dataset,
                                              batch_size=batch_size,
                                              pin_memory=True)
model = torch.load("models/final_model.pth")
model = model.to(device)

for i in range(7):
    train_num, train_acc = type_acc(train_dataloader, i)
    #vali_num, vali_acc = type_acc(vali_dataloader, i)
    print("The training accuracy of {} {} pictures is {}".format(train_num, emotion[i], train_acc))
    #print("The validation accuracy of {} {} pictures is {}.".format(vali_num, emotion[i], vali_acc))

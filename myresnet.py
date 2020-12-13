import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class resBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(resBlock, self).__init__()

        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channel)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        out = self.stage(X)
        out = out + self.shortcut(X)
        out = self.relu(out)

        return out



class ResModel(nn.Module):
    def __init__(self, n_class):
        super(ResModel, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(resBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(resBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(resBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(resBlock, 512, 2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, n_class)


    def make_layer(self, block, channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channel, stride))
            self.inchannel = channel
        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.conv1(X)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        #out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

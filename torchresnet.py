import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class torchResnet18(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0):
        super(torchResnet18, self).__init__()
        self.drop_rate = drop_rate

        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_features = list(resnet.children())[-1].in_features

        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)

        out = self.fc(x)
        return out

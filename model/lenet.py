# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019-09-18 12:38


import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5_Mnist(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_Mnist, self).__init__()

        self.c1 = nn.Conv2d(1, 20, kernel_size=(5, 5), padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc5 = nn.Linear(1250, 320)
        self.fc6 = nn.Linear(320, num_classes)

    def forward(self, img):
        output = self.c1(img)
        output = F.relu(output)
        output = self.s2(output)

        output = self.c3(output)
        output = F.relu(output)
        output = self.s4(output)

        output = output.view(img.size(0), -1)

        output = self.fc5(output)
        output = F.relu(output)

        output = self.fc6(output)
        return output


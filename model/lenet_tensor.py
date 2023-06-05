# -*- coding: UTF-8 -*-

# Author: Perry
# @Time: 2019/12/3 23:09


import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.tensor_conv import TensorRingConvolution
from .layers.tensor_fc import TensorRingLinear



class LeNet5_Mnist_Tensor(nn.Module):
    def __init__(self, num_classes=10, RANK=None):
        super(LeNet5_Mnist_Tensor, self).__init__()

        # self.c1 = nn.Conv2d(1, 20, kernel_size=(5, 5), padding=2)
        # self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.c3 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        # self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #
        # self.fc5 = nn.Linear(1250, 320)
        # self.fc6 = nn.Linear(320, num_classes)

        rank = 5
        self.c1 = TensorRingConvolution([RANK[0], RANK[1]], [RANK[2], RANK[3]], [1], [4, 5], 5, padding=2)
        # self.c1 = TensorRingConvolution([2, 3], [4, 5], [1], [4, 5], 5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = TensorRingConvolution([RANK[4], RANK[5], RANK[6]], [RANK[7], RANK[8]], [4, 5], [5, 10], kernel_size=5)
        # self.c3 = TensorRingConvolution([2, 3, 4], [5, 6, 7], [4, 5], [5, 5, 2], kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # self.fc5 = nn.Linear(1250, 320)
        # self.fc6 = nn.Linear(320, num_classes)
        self.fc5 = TensorRingLinear(1250, 320, [5, 5, 5, 10], [5, 8, 8], [RANK[9], RANK[10], RANK[11], RANK[12], RANK[13], RANK[14], RANK[15]])
        self.fc6 = TensorRingLinear(320, num_classes, [5, 8, 8], [10], [RANK[16], RANK[17], RANK[18], RANK[19]])

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


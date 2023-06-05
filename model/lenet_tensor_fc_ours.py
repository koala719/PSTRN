# -*- coding: UTF-8 -*-

# Author: Perry
# @Time: 2019/12/3 23:09


import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.tensor_conv import TensorRingConvolution
from .layers.tensor_fc import TensorRingLinear



class LeNet5_Mnist_Tensor_NANNAN_OURS(nn.Module):
    def __init__(self, num_classes=10, RANK=None, init="ours_lenet"):
        super(LeNet5_Mnist_Tensor_NANNAN_OURS, self).__init__()

        # self.c1 = nn.Conv2d(1, 20, kernel_size=(5, 5), padding=2)
        # self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.c3 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        # self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #
        # self.fc5 = nn.Linear(1250, 320)
        # self.fc6 = nn.Linear(320, num_classes)

        Rank = RANK
        # Rank = [1, 1, 4, 3, 4, 4, 4, 5, 4, 5, 5, 4, 5, 5, 2, 5, 4, 5, 5, 6]
        self.c1 = TensorRingConvolution([Rank[0], Rank[1]], [Rank[2], Rank[3]], [1], [4, 5], 5, padding=2, init=init)
        # self.c1 = TensorRingConvolution([2, 3], [4, 5], [1], [4, 5], 5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = TensorRingConvolution([Rank[4], Rank[5], Rank[6]], [2, Rank[8]], [4, 5], [5, 10], kernel_size=5, init=init)
        # self.c3 = TensorRingConvolution([2, 3, 4], [5, 6, 7], [4, 5], [5, 5, 2], kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.compression_rat2 = self.c1.compression_rat
        self.compression_rat3 = self.c3.compression_rat

        # self.fc5 = nn.Linear(1250, 320)
        # self.fc6 = nn.Linear(320, num_classes)
        self.fc5 = TensorRingLinear(1250, 320, [5, 5, 5, 10], [5, 8, 8], [RANK[9], Rank[10], Rank[11], Rank[12], Rank[13], Rank[14], Rank[15]], init=init)
        self.fc6 = TensorRingLinear(320, num_classes, [5, 8, 8], [10], [Rank[16], Rank[17], Rank[18], Rank[19]], init=init)

        self.compression_rat0 = self.fc5.compression_rat
        self.compression_rat1 = self.fc6.compression_rat
        self.compression_rat = min(self.compression_rat0, self.compression_rat1, self.compression_rat2, self.compression_rat3)
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


class LeNet5_Fashion_Mnist_Tensor_OURS(nn.Module):
    def __init__(self, num_classes=10, RANK=None, init="ours_resnet"):
        super(LeNet5_Fashion_Mnist_Tensor_OURS, self).__init__()

        # Rank = [1, 1, 1, 1, 1, 1, 1]
        # self.model1=nn.Sequential(nn.Conv2d(1, 20, kernel_size=(5, 5), padding=2),
        #                           nn.Relu(),
        #                           nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #                           nn.Conv2d(20, 50, kernel_size=(5, 5)),
        #                           nn.ReLU(),
        #                           nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.c1 = nn.Conv2d(1, 20, kernel_size=(5, 5), padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc5 = TensorRingLinear(1250, 320, [5, 5, 5, 10], [5, 8, 8], [RANK[0], RANK[1], RANK[2], RANK[3], RANK[4], RANK[5], RANK[6]], init=init)
        self.fc6 = nn.Linear(320, num_classes)
        self.compression_rat = self.fc5.compression_rat

    def forward(self, img):
        output = self.c1(img)
        output = F.relu(output)
        output = self.s2(output)

        output = self.c3(output)
        output = F.relu(output)
        output = self.s4(output)
        # output = self.model1(img)

        output = output.view(img.size(0), -1)

        output = self.fc5(output)
        output = F.relu(output)

        output = self.fc6(output)
        return output

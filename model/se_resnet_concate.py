# -*- coding: UTF-8 -*-

# Author: Perry
# @Time: 2019/10/28 20:37


import torch
import torch.nn as nn

import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, stride=1, reduction=16):
        super().__init__()

        self.conv_z1 = nn.Conv2d(inp_dim, oup_dim, kernel_size=1, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(inp_dim, oup_dim, 1, 1, padding=0)
        self.conv_n1 = nn.Conv2d(inp_dim, oup_dim, 1, 1, padding=0)

        self.conv_z2 = nn.Conv2d(inp_dim * 2, oup_dim, 3, 1, padding=1, groups=oup_dim)
        self.conv_r2 = nn.Conv2d(inp_dim * 2, oup_dim, 3, 1, padding=1, groups=oup_dim)
        self.conv_n2 = nn.Conv2d(inp_dim * 2, oup_dim, 3, 1, padding=1, groups=oup_dim)

        self.se = SELayer(oup_dim, reduction)
        self.relu = nn.LeakyReLU(0.2)
        print('first Depthwise, then Pointwise, X and H concate')

    def forward(self, x, h=None):
        if h is None:
            # z = torch.sigmoid(self.conv_xz(x))
            # f = torch.tanh(self.conv_xn(x))
            # h = z * f
            h = x
        else:
            i = torch.cat([x, h], 1)
            z = torch.sigmoid(self.conv_z1(self.conv_z2(i)))
            r = torch.sigmoid(self.conv_r1(self.conv_r2(i)))
            j = r * h
            n = torch.tanh(self.conv_n1(self.conv_n2(torch.cat([x, j], 1))))
            h = (1 - z) * h + z * n

        # h = self.relu(self.se(h))
        # return h, h
        h = self.se(h)
        return h, h

class SEBottleneckGRU(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, downsample_h=None, reduction=16, rnn=None, type=1, concat = False):
        super(SEBottleneckGRU, self).__init__()
        self.type = type
        print('type is:', type)
        self.rnn = rnn
        self.concat = concat
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if self.concat == True:
            print('Concate')
            self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=1, bias=False)

        else:
            self.conv_fuse = nn.Conv2d(planes * 2, planes, 1, 1, 0)
            self.bn_fuse = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.bn_rnnout = nn.BatchNorm2d(planes)
        self.bn_h = nn.BatchNorm2d(planes)
        self.stride = stride
        self.downsample_h = downsample_h

    def forward(self, x):
        if len(x) == 2:
            x, h = x
        else:
            h = None
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out2 = self.conv2(out)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        if self.type == 1:
            rnn_in = out
            if self.downsample_h is not None:
                rnn_in = self.downsample_h(rnn_in)
            else:
                rnn_in = rnn_in
        else:
            rnn_in = out2


        rnn_out, h = self.rnn(rnn_in, h)
        rnn_out = self.relu(self.bn_rnnout(rnn_out))
        h = self.relu(self.bn_h(h))
        out = torch.cat([out2, rnn_out], 1)
        if self.concat == False:
            out = self.relu(self.bn_fuse(self.conv_fuse(out)))

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, h

class ResNetGRU(nn.Module):

    def __init__(self, block, layers, kernel = 3, num_classes=1000, para_reduce=False, type=1, concat=False):
        self.inplanes = 64
        super(ResNetGRU, self).__init__()
        self.para_reduce = para_reduce
        self.concat = concat
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.rnn_kernel = kernel
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rnn1 = ConvGRU(inp_dim=64, oup_dim=64, kernel=self.rnn_kernel)
        self.rnn2 = ConvGRU(128, 128, self.rnn_kernel)
        self.type = type
        if layers[2] > 10:
            self.rnn3 = None
        else:
            self.rnn3 = ConvGRU(256, 256, self.rnn_kernel)

        if self.para_reduce == True:
            self.rnn4 = None
        else:
            self.rnn4 = ConvGRU(512, 512, self.rnn_kernel)


        print("prcessing 1...")
        self.layer1 = self._make_layer(block, 64, layers[0], rnn=self.rnn1)
        print("prcessing 2...")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rnn=self.rnn2)
        print("prcessing 3...")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rnn=self.rnn3)
        print("prcessing 4...")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, rnn=self.rnn4)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, rnn=None):
        downsample = None
        downsample_h = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        if stride != 1 and rnn is not None:
            downsample_h = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                          padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU()
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, downsample_h=downsample_h, rnn=rnn, type=self.type, concat=self.concat))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rnn=rnn, type=self.type, concat=self.concat))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, h = self.layer1(x)
        x, h = self.layer2(x)
        if self.rnn3 is None:
            x = self.layer3(x)
        else:
            x, h = self.layer3(x)

        if self.rnn4 is None:
            x = self.layer4(x)
        else:
            x, h = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnet50GRU(para_reduce, kernel, num_classes, type=1, concat = False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetGRU(SEBottleneckGRU, [3, 4, 6, 3], kernel = kernel, num_classes=num_classes, para_reduce=para_reduce, type=type, concat=concat)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50GRU_type1_DP_concate(num_classes=1000):
    return se_resnet50GRU(para_reduce=False, kernel=1, num_classes=num_classes, type=1, concat=True)


def se_resnet50GRU_type2_DP_concate(num_classes=1000):
    return se_resnet50GRU(para_reduce=False, kernel=1, num_classes=num_classes, type=2, concat=True)
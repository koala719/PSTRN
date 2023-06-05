# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019-12-03 18:28

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class TensorRingConvolution(torch.nn.Module):
    def __init__(self, input_rank, output_rank, input_shape, output_shape, kernel_size, stride=1, padding=0, bias=True, init="ours_resnet"):
        super(TensorRingConvolution, self).__init__()

        assert len(input_rank) == len(input_shape) + 1
        assert len(output_rank) == len(output_shape)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.input_rank = np.array(input_rank)
        self.input_shape = np.array(input_shape)
        self.input_size = np.prod(self.input_shape)

        self.output_rank = np.array(output_rank)
        self.output_rank_complement = np.append(output_rank, self.input_rank[0])
        self.output_shape = np.array(output_shape)
        self.output_size = np.prod(self.output_shape)

        self.kernel_channel_in = self.input_rank[-1]
        self.kernel_channel_out = self.output_rank[0]
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.input_num = len(self.input_shape)
        self.output_num = len(self.output_shape)
        self.nodes_num = self.input_num + self.output_num

        self.weights = []
        node_shapes = self.generate_node_shapes()
        for node_shape in node_shapes:
            tmp = nn.Parameter(torch.Tensor(node_shape["shape"][0], node_shape["shape"][1]))
            self.weights.append(tmp)
            self.register_parameter(node_shape["name"], tmp)
        self.kernel = nn.Conv2d(self.kernel_channel_in, self.kernel_channel_out,
                                self.kernel_size, self.stride, self.padding, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.compression_rat = self.calculate_compression()

        self.init = init
        if self.init == "ours_resnet":
            self.reset_parameters_ours_resnet()
        elif self.init == "ours_lenet":
            self.reset_parameters_ours_lenet()
        elif "normal" in self.init:
            self.reset_parameters_normal(self.init)
        elif self.init == "kaiming":
            self.reset_parameters_kaiming()
        elif self.init == "xavier":
            self.reset_parameters_xavier()
        else:
            raise KeyError("The initialization %s is not existed." % self.init)

    def forward(self, inputs):
        res = self.tensor_contract(inputs)
        if self.bias is not None:
            # fused op is marginally faster
            res = torch.add(self.bias, res)

        # res: [b, H, W, O1O2O3O4] | res: [b, O1O2O3O4, H, W]
        res = res.permute(0, 3, 1, 2)
        res = res.contiguous()

        return res

    def reset_parameters_normal(self, init):
        std = float("0." + init.split("_")[1])
        for weight in self.parameters():
            nn.init.normal_(weight.data, 0, std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_resnet(self):
        nn.init.normal_(self.weights[0], std=math.sqrt(1./self.input_size))
        # nn.init.normal_(self.weights[0], std=math.sqrt(2./(self.input_size + self.input_rank[0]*self.input_rank[-1])))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self.weights[i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self.weights[self.input_num], std=math.sqrt(math.sqrt(2.)/(self.input_rank[0]*self.output_rank[0])))
        # nn.init.normal_(self.weights[self.input_num], std=math.sqrt(4./(self.output_size + self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self.weights[self.input_num+i], mode="fan_in", nonlinearity="linear")

        nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="linear")
        # nn.init.xavier_normal_(self.kernel.weight.data)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_lenet(self):
        nn.init.normal_(self.weights[0], std=math.sqrt(1./self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self.weights[i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self.weights[self.input_num], std=math.sqrt(1./(self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self.weights[self.input_num+i], mode="fan_in", nonlinearity="linear")

        nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="relu")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # def reset_parameters_kaiming(self):
    #     for weight in self.weights[:-1]:
    #         nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="linear")
    #     nn.init.kaiming_normal_(self.weights[-1], mode="fan_in", nonlinearity="relu")
    #
    #     nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="linear")
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def reset_parameters_kaiming(self):
        nn.init.normal_(self.weights[0], std=math.sqrt(2./self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self.weights[i], mode="fan_in", nonlinearity="relu")

        nn.init.normal_(self.weights[self.input_num], std=math.sqrt(2./(self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self.weights[self.input_num+i], mode="fan_in", nonlinearity="relu")

        nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="relu")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # def reset_parameters_xavier(self):
    #     for weight in self.weights:
    #         nn.init.xavier_normal_(weight)
    #
    #     nn.init.xavier_normal_(self.kernel.weight.data)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def reset_parameters_xavier(self):
        nn.init.normal_(self.weights[0], std=math.sqrt(2./(self.input_size + self.input_rank[0]*self.input_rank[-1])))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self.weights[i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self.weights[self.input_num], std=math.sqrt(2./(self.output_size + self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self.weights[self.input_num+i], mode="fan_in", nonlinearity="linear")

        nn.init.xavier_normal_(self.kernel.weight.data)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def generate_node_shapes(self):
        node_shapes = []
        for i in range(self.nodes_num):
            if i < self.input_num:
                left_rank = self.input_rank[i]
                right_rank = self.input_rank[i + 1]
                middle_rank = self.input_shape[i]

                if i == 0:
                    tmp = dict(
                        name="input_node%d" % i,
                        shape=(left_rank * middle_rank, right_rank)
                    )
                else:
                    tmp = dict(
                        name="input_node%d" % i,
                        shape=(middle_rank * right_rank, left_rank)
                    )

            else:
                output_i = i - self.input_num

                left_rank = self.output_rank_complement[output_i]
                right_rank = self.output_rank_complement[output_i + 1]
                middle_rank = self.output_shape[output_i]

                if output_i == 0:
                    tmp = dict(
                        name="output_node%d" % output_i,
                        shape=(left_rank * middle_rank, right_rank)
                    )
                else:
                    tmp = dict(
                        name="output_node%d" % output_i,
                        shape=(middle_rank * right_rank, left_rank)
                    )
            node_shapes.append(tmp)
        return node_shapes

    def tensor_contract(self, inputs):
        batch_size = inputs.shape[0]
        image_hw = inputs.shape[2:]
        res = inputs.view(batch_size, -1, *image_hw)
        # res: [b, I0, I1, I2, H, W] | res: [b, H, W, I0, I1, I2]
        res = res.permute(0, 2, 3, 1)

        I_in = self.weights[0]
        for i in range(1, self.input_num):
            weight_tmp = self.weights[i]
            left_rank = self.input_rank[i]

            # I_in: [r0I0, r1], w: [I1r2, r1] | I_in: [r0I0, I1r2]
            I_in = I_in.reshape(-1, left_rank)
            I_in = F.linear(I_in, weight_tmp)

        # res: [b, H, W, I0, I1, I2], I_in: [r0I0I1, I2r3] | res: [bHW, r0r3]
        I_in = I_in.reshape(self.input_rank[0], -1, self.input_rank[-1])
        I_in = I_in.permute(0, 2, 1)
        I_in = I_in.reshape(-1, self.input_size)
        # res = res.reshape(-1, self.input_size)
        res = F.linear(res, I_in)

        # res: [bHW, r0r3] | res: [br0, r3, H, W]
        res = res.reshape(batch_size, -1, self.input_rank[0]*self.input_rank[-1])
        res = res.permute(0, 2, 1)
        res = res.reshape(-1, self.input_rank[-1], *image_hw)

        #### Dropout

        # res: [bHW, r0r3] | res: [br0, r4, nH, nW]
        res = self.kernel(res)
        image_new_hw = res.shape[2:]
        # res: [br0, r4, nH, nW] | res: [b, nHnW, r0r4]
        res = res.reshape(batch_size, self.input_rank[0]*self.output_rank[0], -1)
        res = res.permute(0, 2, 1)

        ##### Dropout

        O_out = self.weights[self.input_num]
        for i in range(1, self.output_num):

            weight_tmp = self.weights[self.input_num + i]
            left_rank = self.output_rank_complement[i]

            # O_out: [r4O0, r5], w: [O1r6, r5] | O_out: [r4O0, O1r6]
            O_out = O_out.reshape(-1, left_rank)
            O_out = F.linear(O_out, weight_tmp)

        # O_out: [r4O0O1, O2r0]| O_out: [O0O1O2, r0r4]
        O_out = O_out.reshape(self.output_rank[0], -1, self.input_rank[0])
        O_out = O_out.permute(1, 2, 0)

        # res: [b, nHnW, r0r4], O_out: [O0O1O2, r0r4]] | res: [bnHnW, O0O1O2]
        O_out = O_out.reshape(-1, self.input_rank[0]*self.output_rank[0])
        res = F.linear(res, O_out)
        res = res.reshape(batch_size, *image_new_hw, -1)

        return res

    def calculate_compression(self):
        param_origin = self.input_size * self.output_size * np.prod(self.kernel_size)

        param_input = np.sum(self.input_rank[:-1] * self.input_shape * self.input_rank[1:])
        param_kernel = np.prod(self.kernel_size) * self.kernel_channel_in * self.kernel_channel_out
        param_output = np.sum(self.output_rank_complement[:-1] * self.output_shape * self.output_rank_complement[1:])
        param_tr = param_input + param_kernel + param_output

        compression_ration = param_origin / param_tr
        print("compression_ration is: ", compression_ration)
        return compression_ration

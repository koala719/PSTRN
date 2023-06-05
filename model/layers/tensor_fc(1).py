# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019-12-03 18:26


# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


import numpy as np

import math


class TensorRingLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, input_shape: list,
                 output_shape: list, rank_shape: list, bias: bool = True):
        super(TensorRingLinear, self).__init__()

        # The size of the original matrix
        self.input_size = input_size
        self.output_size = output_size

        # The shape of the tensor ring decomposition
        self.input_shape = np.array(input_shape)
        self.output_shape = np.array(output_shape)
        self.rank_shape = np.array(rank_shape)

        # Check whether shapes are right
        self.check_shape_setting()

        self.nodes_num = len(self.rank_shape)
        self.input_num = len(self.input_shape)
        self.output_num = len(self.output_shape)

        assert self.input_num + self.output_num == self.nodes_num

        self.tr_ranks_line = np.append(self.rank_shape, self.rank_shape[0])
        self.whole_node_shape = np.append(self.input_shape, self.output_shape)

        self.weights = []
        node_shapes = self.generate_node_shapes()
        for node_shape in node_shapes:
            tmp = nn.Parameter(torch.Tensor(node_shape["shape"][0], node_shape["shape"][1]))
            self.weights.append(tmp)
            self.register_parameter(node_shape["name"], tmp)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.calculate_compression()
        self.reset_parameters()

    def forward(self, inputs):
        res = self.tensor_contract(inputs, self.weights)
        if self.bias is not None:
            # fused op is marginally faster
            res = torch.add(self.bias, res)
        return res

    # def reset_parameters(self):
    #     for weight in self.parameters():
    #         nn.init.normal_(weight.data, 0, 0.233)
    #
    #     if self.bias is not None:
    #         nn.init.normal_(self.bias, 0,  0.01)
    #         # nn.init.zeros_(self.bias)

    def reset_parameters(self):
        for weight in self.weights[:-1]:
            nn.init.kaiming_normal_(weight, a=0, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.weights[-1], a=0, mode="fan_in", nonlinearity="relu")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # def reset_parameters(self):
    #     param_origin = self.input_size * self.output_size
    #     d = len(self.input_shape) + len(self.output_shape)
    #     for weight in self.weights:
    #         nn.init.normal_(weight.data, 0, (2/10)**(1/d)*1/(self.tr_ranks_line[0]**(1/2)))
    #
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    # def reset_parameters(self):
    #     for weight in self.weights:
    #         nn.init.xavier_normal_(weight)
    #
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    # To avoid forgetting to design shapes wrongly
    def check_shape_setting(self):
        assert self.input_size == np.prod(self.input_shape), "The decomposition of the input_size is not suitable!"
        assert self.output_size == np.prod(
            self.output_shape), "The decomposition of the output_size is not suitable!"
        # print("The Tensor Ring shape is qualified!")

    def generate_node_shapes(self):
        node_shapes = []
        for i in range(self.nodes_num):
            left_rank = self.tr_ranks_line[i]
            right_rank = self.tr_ranks_line[i + 1]
            middle_rank = self.whole_node_shape[i]

            if i == 0:
                tmp = dict(
                    name="node%d" % i,
                    shape=(left_rank * right_rank, middle_rank)
                )
            elif i < self.input_num:
                tmp = dict(
                    name="node%d" % i,
                    shape=(right_rank, middle_rank * left_rank)
                )
            elif i < self.nodes_num - 1:
                tmp = dict(
                    name="node%d" % i,
                    shape=(right_rank * middle_rank, left_rank)
                )
            else:
                tmp = dict(
                    name="node%d" % i,
                    shape=(middle_rank, right_rank * left_rank)
                )
            node_shapes.append(tmp)
        return node_shapes

    def tensor_contract(self, inputs, weights):
        batch_size = inputs.shape[0]
        res = inputs.view(-1, *self.input_shape.tolist())
        # res: [b, I0, I1, I2] | res: [b, I2, I1, I0]
        trans_tmp = [0]
        trans_tmp.extend([self.input_num - i for i in range(self.input_num)])
        res = res.permute(*trans_tmp)
        for i in range(self.nodes_num):
            weight_tmp = weights[i]
            weight_shape_right = weight_tmp.shape[1]

            left_rank = self.tr_ranks_line[i]
            right_rank = self.tr_ranks_line[i + 1]

            if i == 0:
                # res: [b, I2, I1, I0], w: [I0, r1r0] | res: [b, r0, I2, I1, r1]
                res = res.reshape(-1, weight_shape_right)
                # res = res.view(-1, weight_rec)
                res = F.linear(res, weight_tmp)
                res = res.view(batch_size, -1, left_rank)
                res = res.permute(0, 2, 1)
            elif i < self.input_num:
                # res: [b, r0, I2, I1, r1], w: [I1r1, r2] | res: [b, r0, I2, r2]
                res = res.reshape(-1, weight_shape_right)
                res = F.linear(res, weight_tmp)
            elif i < self.nodes_num - 1:
                # res: [b, r0, r3], w: [r3, O1r4] | res: [b, r0, O1r4]
                res = res.reshape(-1, weight_shape_right)
                res = F.linear(res, weight_tmp)
            else:
                # res: [b, r0, O1O2O3r6], w: [r6r0, O4] | res: [b, O1O2O3O4]
                res = res.view(batch_size, right_rank, -1, left_rank)
                res = res.permute(0, 2, 1, 3)
                res = res.reshape(-1, weight_shape_right)
                res = F.linear(res, weight_tmp)
                res = res.view(batch_size, -1)
        return res

    def calculate_compression(self):
        param_origin = self.input_size * self.output_size
        param_tr = np.sum(self.tr_ranks_line[:-1] * self.tr_ranks_line[1:] * self.whole_node_shape)
        compression_ration = param_origin / param_tr
        print("compression_ration is: ", compression_ration)
        return compression_ration


class BTTuncker_FC(nn.Module):
    def __init__(self, input_size, output_size, input_shape, output_shape, rank_shape,  block_num):
        super(BTTuncker_FC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.input_shape = np.array(input_shape)
        self.output_shape = np.array(output_shape)
        self.rank_shape = np.array(rank_shape)
        self.block_num = block_num

        self.check_dec_init()
        self.define_info()

        self.weights = []
        blocks = self.dec_split()
        for block in blocks:
            core = block["core"]
            factors = block["factors"]

            core_param_tmp = nn.Parameter(torch.Tensor(*core["shape"]))
            setattr(self, core["name"], core_param_tmp)

            factor_tmp = []
            for factor in factors:
                factor_param_tmp = nn.Parameter(torch.Tensor(*factor["shape"]))
                setattr(self, factor["name"], factor_param_tmp)
                factor_tmp.append(factor_param_tmp)
            block_tmp = dict(
                core=core_param_tmp,
                factors=factor_tmp
            )
            self.weights.append(block_tmp)

        self.cal_compression()
        self.reset_parameters()

    def forward(self, inputs):
        res = self.dec_multy(inputs, self.weights)
        return res

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.output_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # To avoid forgetting to initial these variables
    def check_dec_init(self):
        assert self.input_size == np.prod(self.input_shape), "The decomposition of the input_size is not suitable!"
        assert self.output_size == np.prod(
            self.output_shape), "The decomposition of the output_size is not suitable!"
        print("All to-be-inited variable has been inited!")

    def define_info(self):
        self.factor_num = len(self.rank_shape)
        self.input_num = len(self.input_shape)
        self.output_num = len(self.output_shape)

        self.core_size = np.prod(self.rank_shape)

        assert self.input_num == self.output_num == self.factor_num, "Factors is not equal!"

    def dec_split(self):
        blocks = []

        for i in range(self.block_num):
            core_tmp = dict(
                name="core_block%d" % i,
                shape=tuple(self.rank_shape.tolist())
            )
            factors_tmp = []
            for j in range(self.factor_num):
                factor_tmp = dict(
                    name="factor%d_block%d" % (j, i),
                    shape=(self.input_shape[j], self.output_shape[j], self.rank_shape[j])
                )
                factors_tmp.append(factor_tmp)
            block_tmp = dict(
                core=core_tmp,
                factors=factors_tmp
            )
            blocks.append(block_tmp)
        return blocks

    def dec_multy(self, inputs, weights):
        """

        :param inputs: The Input Tensor, shape like[batchsize, length]
        :param weights: A dict include key: core, factors
        :return: The results of multiply
        """
        batch_size = inputs.shape[0]
        input_tmp = inputs.view(-1, *self.input_shape.tolist())
        # res: [b, I0, I1, I2] | res: [b, I2, I1, I0]
        trans_tmp = [0]
        trans_tmp.extend([self.input_num - i for i in range(self.input_num)])
        input_tmp = input_tmp.permute(*trans_tmp)

        res = 0
        for weight in weights:
            core = weight["core"]
            factors = weight["factors"]
            factor_offset = 1

            cal_tmp = input_tmp
            for factor in factors:
                I, J, R = factor.shape
                cal_tmp = cal_tmp.reshape(-1, I).matmul(factor.view(I, -1))
                cal_tmp = cal_tmp.view(batch_size*factor_offset, -1, J*R)
                cal_tmp = cal_tmp.permute(0, 2, 1)
                factor_offset *= J

            cal_tmp = cal_tmp.reshape(-1, self.core_size).matmul(core.view(self.core_size, -1))
            cal_tmp = cal_tmp.view(batch_size, -1)
            res += cal_tmp
        return res

    def cal_compression(self):
        param_origin = self.input_size * self.output_size
        param_tr = self.block_num * (np.prod(self.rank_shape) + np.sum(
            self.rank_shape * self.input_shape * self.output_shape
        ))
        compression_ration = param_origin / param_tr
        print("compression_ration is: ", compression_ration)
        return compression_ration

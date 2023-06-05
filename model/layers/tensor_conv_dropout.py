# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019-12-03 18:28

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class TensorRingConvolution(torch.nn.Module):
    def __init__(self, input_rank, output_rank, input_shape, output_shape, kernel_size, stride=1, padding=0, bias=True, is_dropout=True, drop_rate=0.1):
        super(TensorRingConvolution, self).__init__()

        assert len(input_rank) == len(input_shape) + 1
        assert len(output_rank) == len(output_shape)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.input_rank = np.array(input_rank)
        self.input_shape = np.array(input_shape)

        self.output_rank = np.array(output_rank)
        self.output_rank_complement = np.append(
            output_rank, self.input_rank[0])
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
        if is_dropout:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = None
        self.weights = []
        node_shapes = self.generate_node_shapes()
        for node_shape in node_shapes:
            tmp = nn.Parameter(torch.Tensor(*node_shape["shape"]))
            self.weights.append(tmp)
            self.register_parameter(node_shape["name"], tmp)

        self.kernel = nn.Conv2d(self.kernel_channel_in, self.kernel_channel_out,
                                self.kernel_size, self.stride, self.padding, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.calculate_compression()
        self.reset_parameters()

    def forward(self, inputs):
        res = self.tensor_contract(inputs, self.weights, self.kernel)
        if self.bias is not None:
            # fused op is marginally faster
            res = torch.add(self.bias, res)

        # res: [b, H, W, O1O2O3O4] | res: [b, O1O2O3O4, H, W]
        res = res.permute(0, 3, 1, 2)
        res = res.contiguous()

        return res

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.normal_(weight.data, 0, 0.233)

        if self.bias is not None:
            nn.init.normal_(self.bias, 0,  0.01)

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
                        shape=(left_rank * right_rank, middle_rank)
                    )
                else:
                    tmp = dict(
                        name="input_node%d" % i,
                        shape=(right_rank, middle_rank * left_rank)
                    )

            else:
                output_i = i - self.input_num

                left_rank = self.output_rank_complement[output_i]
                right_rank = self.output_rank_complement[output_i + 1]
                middle_rank = self.output_shape[output_i]

                if i < self.nodes_num - 1 or self.dropout is not None:
                    tmp = dict(
                        name="output_node%d" % output_i,
                        shape=(right_rank * middle_rank, left_rank)
                    )
                else:
                    tmp = dict(
                        name="output_node%d" % output_i,
                        shape=(middle_rank, right_rank * left_rank)
                    )
            node_shapes.append(tmp)
        return node_shapes

    def tensor_contract(self, inputs, weights, kernel):
        batch_size = inputs.shape[0]
        image_hw = inputs.shape[2:]
        res = inputs.view(-1, *self.input_shape.tolist(), *image_hw)
        # res: [b, I0, I1, I2, H, W] | res: [b, H, W, I2, I1, I0]
        trans_tmp = [0, self.input_num + 1, self.input_num + 2]
        trans_tmp.extend([self.input_num - i for i in range(self.input_num)])
        res = res.permute(*trans_tmp)
        for i in range(self.input_num):
            weight_tmp = weights[i]
            weight_shape_right = weight_tmp.shape[1]

            left_rank = self.input_rank[i]
            right_rank = self.input_rank[i + 1]

            if i == 0:
                # res: [b, H, W, I2, I1, I0], w: [I0, r1r0] | res: [b r0 H W I2 I1, r1]
                res = res.reshape(-1, weight_shape_right)
                # res = res.view(-1, weight_rec)
                res = F.linear(res, weight_tmp)
                res = res.view(batch_size, -1, left_rank)
                res = res.permute(0, 2, 1)
                res = res.reshape(-1, right_rank)
            else:
                # res: [b r0 H W I2 I1, r1], w: [I1r1, r2] | res: [b r0 H W I2, r2]
                res = res.view(-1, weight_shape_right)
                res = F.linear(res, weight_tmp)
            if self.dropout is not None:
                dropout_matrix = torch.diag(
                    self.dropout(torch.ones(right_rank, device='cuda')))
                res = F.linear(res, dropout_matrix)
        # res: [b, r0, H, W, r3], kernel: [r3, r4] | res: [br0, r3, H, W]
        res = res.reshape(-1, *image_hw, self.input_rank[-1])
        res = res.permute(0, 3, 1, 2)

        # Dropout

        res = kernel(res)
        image_new_hw = res.shape[2:]
        res = res.view(batch_size, self.input_rank[0], self.output_rank[0], -1)
        res = res.permute(0, 1, 3, 2)

        # Dropout

        for i in range(self.output_num):

            weight_tmp = weights[self.input_num + i]
            weight_shape_right = weight_tmp.shape[1]

            left_rank = self.output_rank_complement[i]
            right_rank = self.output_rank_complement[i + 1]

            if i + 1 < self.output_num or self.dropout is not None:
                res = res.reshape(-1, weight_shape_right)
                # res: [b r0 H W, r3], w: [r3, O1r4] | res: [b r0 H W, O1r4]
                if self.dropout is not None:
                    dropout_matrix = torch.diag(
                        self.dropout(torch.ones(left_rank, device='cuda')))
                    res = F.linear(res, dropout_matrix)
                res = F.linear(res, weight_tmp)

            if i + 1 == self.output_num:
                if self.dropout is None:
                    # res: [b, r0, H, W, O1O2O3r6], w: [r6r0, O4] | res: [b, H, W, O1O2O3O4]
                    res = res.view(batch_size, right_rank, -1, left_rank)
                    res = res.permute(0, 2, 1, 3)
                    res = res.reshape(-1, weight_shape_right)
                    res = F.linear(res, weight_tmp)
                    res = res.view(batch_size, *image_new_hw, -1)
                else:
                    # res: [b r0 H W, O1O2O3r0]
                    res = res.view(batch_size, right_rank, -1, right_rank)
                    res = res.permute(0, 2, 1, 3)
                    res = res.reshape(-1, right_rank * right_rank)
                    # res: [b, H, W, O1O2O3, r0 r0], w: [r0 r0, 1] | res : [b, H, W, O1O2O3]
                    dropout_matrix = torch.diag(
                        self.dropout(torch.ones(right_rank, device='cuda')))
                    dropout_vector = dropout_matrix.reshape(1, -1)
                    res = F.linear(res, dropout_vector)
                    res = res.view(batch_size, *image_new_hw, -1)

        return res

    def calculate_compression(self):
        param_origin = np.prod(self.input_shape) * np.prod(self.output_shape)

        param_input = np.sum(
            self.input_rank[:-1] * self.input_shape * self.input_rank[1:])
        param_kernel = np.prod(self.kernel_size) * \
            self.kernel_channel_in * self.kernel_channel_out
        param_output = np.sum(
            self.output_rank_complement[:-1] * self.output_shape * self.output_rank_complement[1:])
        param_tr = param_input + param_kernel + param_output

        compression_ration = param_origin / param_tr
        print("compression_ration is: ", compression_ration)
        return compression_ration

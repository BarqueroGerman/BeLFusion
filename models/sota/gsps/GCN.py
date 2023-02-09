#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, node_n=48, act_f=nn.Tanh(), p_dropout=0, is_bn=False):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_bn = is_bn

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=not is_bn)
        if is_bn:
            self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=not is_bn)
        if is_bn:
            self.bn2 = nn.BatchNorm1d(node_n * in_features)

        # self.do = nn.Dropout(p_dropout)
        self.act_f = act_f

    def forward(self, x):
        y = self.gc1(x)
        if self.is_bn:
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        # y = self.do(y)

        y = self.gc2(y)
        if self.is_bn:
            b, n, f = y.shape
            y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        # y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout=0, num_stage=1, node_n=48, is_bn=False,
                 act_f=nn.Tanh()):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.is_bn = is_bn

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        if is_bn:
            self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        gcbs = []
        for i in range(num_stage):
            gcbs.append(GC_Block(hidden_feature, node_n=node_n, is_bn=is_bn, act_f=act_f))

        self.gcbs = nn.Sequential(*gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        # self.do = nn.Dropout(p_dropout)
        self.act_f = act_f

    def forward(self, x):
        y = self.gc1(x)
        if self.is_bn:
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        # y = self.do(y)
        y = self.gcbs(y)
        y = self.gc7(y)
        y = y + x
        return y


class GCNParts(nn.Module):
    def __init__(self, input_feature, hidden_feature,
                 parts=[[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12], [13, 14, 15]],
                 p_dropout=0, num_stage=1, node_n=48, is_bn=False,
                 act_f=nn.Tanh()):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCNParts, self).__init__()
        self.num_stage = num_stage
        self.is_bn = is_bn
        self.parts = parts
        self.node_n = node_n

        gcns = []
        pall = []
        for p in parts:
            # pall = pall + p
            # node_n = len(pall)
            gcns.append(GCN(input_feature, hidden_feature, num_stage=num_stage,
                            node_n=node_n, is_bn=is_bn, act_f=act_f))
        self.gcns = nn.ModuleList(gcns)

    def forward(self, x, z):
        """
        x: bs, node_n, feat
        z: bs, parts_n, feat
        """
        y = x.clone()
        pall = []
        for i, p in enumerate(self.parts):
            # pall = pall + p
            zt = z[:, i:i + 1].repeat([1, self.node_n, 1])
            xt = torch.cat([y, zt], dim=-1)
            yt = self.gcns[i](xt)
            y[:, p] = yt[:, p, :x.shape[2]]
        return y

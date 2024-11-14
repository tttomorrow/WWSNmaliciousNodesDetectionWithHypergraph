import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import HypergraphConv  # 导入超图卷积层的实现


# 定义超图卷积层
class HypergraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HypergraphConvLayer, self).__init__()
        self.conv = HypergraphConv(in_channels, out_channels)  # 创建超图卷积层

    def forward(self, x, edge_index, edge_weight):
        # 前向传播函数，输入为节点特征、超边连接和超边权重
        return self.conv(x, edge_index, edge_weight)  # 返回卷积层的输出


# 定义图卷积层
class GraphConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True):
        super(GraphConvolution, self).__init__()
        # 初始化输入和输出特征维度
        self.in_features_v = in_features_v  # 节点特征的输入维度
        self.out_features_v = out_features_v  # 节点特征的输出维度
        self.in_features_e = in_features_e  # 边特征的输入维度
        self.out_features_e = out_features_e  # 边特征的输出维度

        # 初始化权重参数
        self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))  # 权重矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features_v))  # 偏置项
        else:
            self.register_parameter('bias', None)  # 如果不使用偏置项，注册为None
        self.reset_parameters()  # 重置参数
        self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, 3))).float())

    def reset_parameters(self):
        # 使用均匀分布初始化权重和偏置
        stdv = 1. / math.sqrt(self.weight.size(1))  # 计算标准差
        self.weight.data.uniform_(-stdv, stdv)  # 初始化权重
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  # 初始化偏置

    def forward(self, H_v, edge_features, adj_e, T):
        print(self.p.shape)
        multiplier = torch.spmm(T, torch.diag((edge_features @ self.p.t()).t()[0])) @ T.to_dense().t()
        mask = torch.eye(multiplier.shape[0])
        M = mask * torch.ones(multiplier.shape[0]) + (1. - mask) * multiplier
        adjusted_A = torch.mul(M, adj_e.to_dense())
        '''
        print("adjusted_A is ", adjusted_A)
        normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
        print("normalized adjusted A is ", normalized_adjusted_A)
        '''
        # to avoid missing feature's influence, we don't normalize the A
        output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
        if self.bias is not None:
            output = output + self.bias
        return output  # 返回输出

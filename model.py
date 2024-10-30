import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import Linear
from torch_geometric.nn import HypergraphConv, GATConv, GCNConv
from AFF import AFF
from mscam import MS_CAM

class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class HypergraphTemporalModel(nn.Module):
    def __init__(self, num_features, embedding_dim, l, num_head, dropout):
        super(HypergraphTemporalModel, self).__init__()
        print("*** Initializing the HypergraphTemporalModel model ***")
        input_dim = num_features + 1
        print("Initializing user and item hypergraph_conv")
        self.embedding_dim = embedding_dim
        self.layer = l

        # 根据参数选择HGCN层数和GCN；1，2，3，4：相应层数的HGCN；5：一层GCN。
        if self.layer == 5:
            self.gcn_conv = GCNConv(input_dim, self.embedding_dim)
        elif self.layer == 1:
            self.hypergraph_conv1 = HypergraphConv(input_dim, self.embedding_dim)
        elif self.layer == 2:
            self.hypergraph_conv1 = HypergraphConv(input_dim, self.embedding_dim)
            self.hypergraph_conv2 = HypergraphConv(self.embedding_dim, self.embedding_dim)
        elif self.layer == 3:
            self.hypergraph_conv1 = HypergraphConv(input_dim, self.embedding_dim)
            self.hypergraph_conv2 = HypergraphConv(self.embedding_dim, self.embedding_dim)
            self.hypergraph_conv3 = HypergraphConv(self.embedding_dim, self.embedding_dim)
        else:
            self.hypergraph_conv1 = HypergraphConv(input_dim, self.embedding_dim)
            self.hypergraph_conv2 = HypergraphConv(self.embedding_dim, self.embedding_dim)
            self.hypergraph_conv3 = HypergraphConv(self.embedding_dim, self.embedding_dim)
            self.hypergraph_conv4 = HypergraphConv(self.embedding_dim, self.embedding_dim)

        # 循环神经网络
        self.hyper_rnn = nn.RNNCell(self.embedding_dim, self.embedding_dim)
        # self.hypergraph_conv2 = HypergraphConv(32, 1)
        # self.conv = GCNConv(32, 1)

        # 多头注意力特征融合
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.dropout = dropout

        # 原始特征进入GCN
        self.gcn_conv1 = GCNConv(num_features, self.embedding_dim)

        # 原始特征进入循环神经网络
        self.features1 = nn.RNNCell(self.embedding_dim, self.embedding_dim)
        self.liner1 = nn.Linear(num_features, self.embedding_dim)

        # 维数调整
        self.liner = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        # self.input_dim = self.embedding_dim * 4
        # self.features = HypergraphConv(self.input_dim, 4)

        # 特征融合
        # self.aff = AFF(self.embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim,
                                                    kdim=self.embedding_dim,
                                                    vdim=self.embedding_dim,
                                                    num_heads=self.num_head,
                                                    dropout=self.dropout)

        # 分类器
        self.class_classifier = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                              nn.ReLU(),
                                              # nn.Dropout(p=self.dropout),
                                              nn.Linear(self.embedding_dim, 2),
                                              nn.Softmax(dim=1))

    def forward(self, fea, timediffs, hyper_index, l, indice):
        input1 = torch.cat([timediffs.cuda(), fea.cuda()], dim=1).cuda()
        # 根据参数选择HGCN层数和GCN；1，2，3，4：相应层数的HGCN；5：一层GCN。
        if l == 5:
            x = self.gcn_conv(input1, hyper_index)
        elif l == 1:
            x = self.hypergraph_conv1(input1, hyper_index.cuda())
        elif l == 2:
            x = self.hypergraph_conv1(input1, hyper_index.cuda())
            x = self.hypergraph_conv2(x, hyper_index.cuda())
        elif l == 3:
            x = self.hypergraph_conv1(input1, hyper_index.cuda())
            x = self.hypergraph_conv2(x, hyper_index.cuda())
            x = self.hypergraph_conv3(x, hyper_index.cuda())
        else:
            x = self.hypergraph_conv1(input1, hyper_index.cuda())
            x = self.hypergraph_conv2(x, hyper_index.cuda())
            x = self.hypergraph_conv4(x, hyper_index.cuda())
            x = self.hypergraph_conv4(x, hyper_index.cuda())

        # 归一化
        # x = F.normalize(x)
        # x = F.relu(x)
        # 循环神经网络
        # print(len(indice))
        x = x[indice]
        # print(x.size())
        # x = self.hyper_rnn(x)
        # x = self.hypergraph_conv2(x, hyper_index.cuda())
        # 归一化
        # x = F.normalize(x)

        # 处理原始特征
        # fea = self.gcn_conv1(fea.cuda(), hyper_index)
        fea = self.liner1(fea.cuda())
        fea = self.features1(fea.cuda())
        # fea = self.liner1(fea.cuda())

        # 特征融合
        # 直接拼接
        input = torch.cat([x.cuda(), fea.cuda()], dim=1).cuda()
        input = self.liner(input.cuda())
        # shared_feature = input

        # 多头注意力特征融合
        shared_feature, _ = self.multihead_attn(input, input, input)
        # shared_feature = torch.cat([x.cuda(), x.cuda()], dim=1).cuda()
        # shared_feature = self.liner(shared_feature.cuda())

        # AFF
        # shared_feature = self.aff(x, fea)

        # 分类结果
        class_output = self.class_classifier(shared_feature.cuda())
        return class_output


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
        input_dim = num_features - 1
        print("Initializing user and item hypergraph_conv")
        self.embedding_dim = embedding_dim
        self.layer = l

        # 根据参数选择HGCN层数和GCN；1，2，3，4：相应层数的HGCN；5：一层GCN。
        if self.layer == 5:
            self.gcn_conv = GCNConv(input_dim, self.embedding_dim)
        elif self.layer == 1:
            self.hypergraph_conv1 = HypergraphConv(input_dim, self.embedding_dim, use_attention=False)
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
        # self.hyper_rnn = nn.RNNCell(self.embedding_dim, self.embedding_dim)
        self.hyper_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        # self.hypergraph_conv2 = HypergraphConv(32, 1)
        # self.conv = GCNConv(32, 1)

        # 多头注意力特征融合
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.dropout = dropout

        # 原始特征进入GCN
        self.gcn_conv1 = GCNConv(num_features - 1, self.embedding_dim)

        # 原始特征进入循环神经网络
        self.features1 = nn.RNNCell(num_features - 1, self.embedding_dim)
        # self.features1 = nn.LSTMCell(num_features, self.embedding_dim)
        self.liner1 = nn.Linear(self.embedding_dim, self.embedding_dim)

        # 维数调整
        self.liner = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.liner2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.liner3 = nn.Linear(input_dim, self.embedding_dim)
        self.liner4 = nn.Linear(num_features - 1, self.embedding_dim)
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
        self.class_classifier = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                                              nn.ReLU(),
                                              nn.Dropout(p=self.dropout),
                                              nn.Linear(self.embedding_dim // 2, 2),
                                              )

    def forward(self, fea, edge_index, edge_attr, edge_weights, l, args):
        device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
        # input1 = torch.cat([timediffs.to(device), fea.to(device)], dim=1)
        input1 = fea[:, 1:].to(device)
        # 根据参数选择HGCN层数和GCN；1，2，3，4：相应层数的HGCN；5：一层GCN。

        if args.weights == 1:
            if l == 5:
                x = self.gcn_conv(input1.to(device), edge_index.to(device), edge_weights.to(device),
                                  edge_attr.to(device))
            elif l == 1:
                x = self.hypergraph_conv1(input1, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv1(input1, edge_index.to(device), edge_weights.to(device))
            elif l == 2:
                x = self.hypergraph_conv1(input1, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv2(x, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
            elif l == 3:
                x = self.hypergraph_conv1(input1, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv2(x, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv3(x, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
            else:
                x = self.hypergraph_conv1(input1, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv2(x, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv4(x, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
                x = self.hypergraph_conv4(x, edge_index.to(device), edge_weights.to(device), edge_attr.to(device))
        else:
            if l == 5:
                x = self.gcn_conv(input1.to(device), edge_index.to(device))
            elif l == 1:
                x = self.hypergraph_conv1(input1, edge_index.to(device))
            elif l == 2:
                x = self.hypergraph_conv1(input1, edge_index.to(device))
                x = self.hypergraph_conv2(x, edge_index.to(device))
            elif l == 3:
                x = self.hypergraph_conv1(input1, edge_index.to(device))
                x = self.hypergraph_conv2(x, edge_index.to(device))
                x = self.hypergraph_conv3(x, edge_index.to(device))
            else:
                x = self.hypergraph_conv1(input1, edge_index.to(device))
                x = self.hypergraph_conv2(x, edge_index.to(device))
                x = self.hypergraph_conv4(x, edge_index.to(device))
                x = self.hypergraph_conv4(x, edge_index.to(device))

        fea = self.liner4(fea[:, 1:].to(device))
        fea = fea.unsqueeze(1)  # 加入时间维度 [batch, seq=1, feat]
        fea, _ = self.hyper_rnn(fea)  # LSTM 输出
        fea = fea.squeeze(1)  # 去掉时间维度恢复 [batch, feat]

        x = x.unsqueeze(1)  # 加入时间维度 [batch, seq=1, feat]
        x, _ = self.hyper_rnn(x)  # LSTM 输出
        x = x.squeeze(1)  # 去掉时间维度恢复 [batch, feat]

        # 特征融合
        # input = torch.cat([fea.to(device), x.to(device)], dim=1).to(device)
        # input = self.liner(input.to(device))
        # shared_feature = input
        shared_feature, _ = self.multihead_attn(fea, x, fea)

        # 分类结果
        class_output = self.class_classifier(shared_feature.to(device))
        return class_output

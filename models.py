import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HypergraphConvLayer, GraphConvolution, CustomBatchNorm




# 定义整体超图模型
class HypergraphModel(nn.Module):
    def __init__(self, num_features, edge_features_size):
        super(HypergraphModel, self).__init__()
        # 初始化两个超图卷积层
        self.layer1 = HypergraphConvLayer(num_features, 32)  # 第一个超图卷积层，输出特征为64
        # self.layer2 = HypergraphConvLayer(64, 32)  # 第二个超图卷积层，输出特征为32
        self.edge_conv = GraphConvolution(32, 32, edge_features_size, 32)  # 图卷积层，融合边特征
        # 分类器
        self.class_classifier = nn.Sequential(nn.Linear(32, 32),
                                              nn.ReLU(),
                                              # nn.Dropout(p=self.dropout),
                                              nn.Linear(32, 2))
        self.norm1 = CustomBatchNorm(32, 32)
        self.norm2 = CustomBatchNorm(edge_features_size, edge_features_size)


    def forward(self, x, edge_index, edge_weight, edge_features, adj, T):
        # 前向传播函数，输入为节点特征、超边连接、超边权重和边特征
        #
        x = self.layer1(x, edge_index, edge_weight)  # 通过第一个超图卷积层
        # x = self.layer2(x, edge_index, edge_weight)  # 通过第二个超图卷积层
        x = self.norm1(x)
        x = F.relu(x)  # 激活函数

        # 聚合节点特征
        node_features = self.aggregate_node_features(x, edge_index)  # 聚合超图卷积得到的节点特征

        edge_features = self.norm2(edge_features)
        shared_feature = self.edge_conv(node_features, edge_features, adj, T)  # 将边特征融合到节点特征中

        # 分类结果
        class_output = self.class_classifier(shared_feature)

        return class_output  # 返回最终输出

    def aggregate_node_features(self, x, edge_index, aggregation_type='mean'):
        num_edges = len(edge_index[1].unique())  # 超边数量

        nodes_features = []  # 用于存储每条超边的特征

        for edge_id in range(num_edges):
            # 获取当前超边连接的所有节点ID
            connected_nodes = edge_index[0, edge_index[1, :] == edge_id]  # 获取所有与当前超边连接的节点

            # 获取这些节点的特征
            edges_features = x[connected_nodes]

            # 根据指定的聚合方式进行处理
            if aggregation_type == 'mean':
                nodes_feature = torch.mean(edges_features, dim=0)
            elif aggregation_type == 'sum':
                nodes_feature = torch.sum(edges_features, dim=0)
            elif aggregation_type == 'max':
                nodes_feature = torch.max(edges_features, dim=0)[0]
            else:
                raise ValueError("Unsupported aggregation type. Use 'mean', 'sum', or 'max'.")

            # 添加该超边的特征到列表
            nodes_features.append(nodes_feature)

        # 将每条超边的特征堆叠成一个矩阵，形状为 (num_nodes, feature_dim)
        stack_nodes_features = torch.stack(nodes_features, dim=0)

        return stack_nodes_features

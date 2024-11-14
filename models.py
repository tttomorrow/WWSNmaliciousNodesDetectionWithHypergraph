import torch
import torch.nn as nn
from layers import HypergraphConvLayer, GraphConvolution


# 定义整体超图模型
class HypergraphModel(nn.Module):
    def __init__(self, num_features, edge_features_size):
        super(HypergraphModel, self).__init__()
        # 初始化两个超图卷积层
        self.layer1 = HypergraphConvLayer(num_features, 64)  # 第一个超图卷积层，输出特征为64
        self.layer2 = HypergraphConvLayer(64, 32)  # 第二个超图卷积层，输出特征为32
        self.edge_conv = GraphConvolution(32, 32, edge_features_size, 32)  # 图卷积层，融合边特征
        # 分类器
        self.class_classifier = nn.Sequential(nn.Linear(32, 32),
                                              nn.ReLU(),
                                              # nn.Dropout(p=self.dropout),
                                              nn.Linear(32, 2),
                                              nn.Softmax(dim=1))

    def forward(self, x, edge_index, edge_weight, edge_features, adj_e, T):
        # 前向传播函数，输入为节点特征、超边连接、超边权重和边特征
        x = self.layer1(x, edge_index, edge_weight)  # 通过第一个超图卷积层
        x = torch.relu(x)  # 激活函数
        x = self.layer2(x, edge_index, edge_weight)  # 通过第二个超图卷积层
        x = torch.relu(x)  # 激活函数
        print('x.shape')
        print(x.shape)

        # 聚合节点特征
        node_features = self.aggregate_node_features(x, edge_index)  # 聚合超图卷积得到的节点特征
        print('node_features.shape')
        print(node_features.shape)
        print('edge_features.shape')
        print(edge_features.shape)
        print('adj_e.shape')
        print(adj_e.shape)
        print('T.shape')
        print(T.shape)
        shared_feature = self.edge_conv(node_features, edge_features, adj_e, T)  # 将边特征融合到节点特征中

        # 分类结果
        class_output = self.class_classifier(shared_feature)

        return class_output  # 返回最终输出

    def aggregate_node_features(self, x, edge_index):
        # 聚合节点特征的函数
        num_nodes = x.size(0)  # 节点数量
        node_features = torch.zeros(num_nodes, x.size(1), device=x.device)  # 初始化聚合后的节点特征

        # 遍历每条超边，进行特征聚合
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]  # 获取超边的源节点和目标节点
            node_features[dst] += x[src]  # 将源节点的特征加到目标节点上

        return node_features / edge_index.size(1)  # 返回平均后的节点特征（可选）

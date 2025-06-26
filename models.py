import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HypergraphConvLayer, GraphConvolution, CustomBatchNorm
from torch_geometric.nn import GATConv, SAGEConv, GCNConv



# 定义整体超图模型
class HypergraphModel(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(HypergraphModel, self).__init__()
        # 初始化两个超图卷积层
        self.layer1 = HypergraphConvLayer(num_features, hidden_size)  # 第一个超图卷积层，输出特征为64
        self.layer2 = HypergraphConvLayer(hidden_size, hidden_size)  # 第二个超图卷积层，输出特征为8
        self.edge_conv = GraphConvolution(hidden_size, hidden_size, edge_features_size, hidden_size)  # 图卷积层，融合边特征
        # 分类器
        self.class_classifier = nn.Sequential(nn.Linear(hidden_size, 16),
                                              nn.ReLU(),
                                              # nn.Dropout(p=0.1),
                                              nn.Linear(16, 2))
        self.norm1 = CustomBatchNorm(num_features, num_features)
        self.norm2 = CustomBatchNorm(edge_features_size, edge_features_size)


    def forward(self, x, edge_index, edge_weight, edge_features, adj, T):
        # 前向传播函数，输入为节点特征、超边连接、超边权重和边特征
        # print(x)
        x = self.norm1(x)
        # print(x)
        x = self.layer1(x, edge_index, edge_weight)  # 通过第一个超图卷积层
        x = self.layer2(x, edge_index, edge_weight)  # 通过第二个超图卷积层

        x = F.relu(x)  # 激活函数

        # 聚合节点特征
        node_features = self.aggregate_node_features(x, edge_index)  # 聚合超图卷积得到的节点特征

        edge_features = self.norm2(edge_features)
        shared_feature = self.edge_conv(node_features, edge_features, adj, T)  # 将边特征融合到节点特征中

        # shared_feature = self.layer2(shared_feature, edge_index, edge_weight)
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



# 定义GAT模型
class GATModel(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(GATModel, self).__init__()
        # 初始化GAT卷积层
        self.gat1 = GATConv(num_features+edge_features_size, hidden_size, heads=4, concat=True, dropout=0.0)
        self.gat2 = GATConv(hidden_size * 4, hidden_size, heads=1, concat=False, dropout=0.1)
        # 分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.norm1 = nn.BatchNorm1d(num_features+edge_features_size)

    def forward(self, x, edge_index, edge_weight, edge_features, adj, T):
        # 特征归一化
        x = self.norm1(x)
        # GAT卷积
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        # 分类
        class_output = self.class_classifier(x)
        return class_output


# 定义GraphSAGE模型
class GraphSAGEModel(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(GraphSAGEModel, self).__init__()
        # 初始化GraphSAGE卷积层
        self.sage1 = SAGEConv(num_features+edge_features_size, hidden_size)
        self.sage2 = SAGEConv(hidden_size, hidden_size)
        # 分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.norm1 = nn.BatchNorm1d(num_features+edge_features_size)

    def forward(self, x, edge_index, edge_weight, edge_features, adj, T):
        # 特征归一化
        x = self.norm1(x)
        # SAGE卷积
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        # x = self.sage2(x, edge_index)
        # 分类
        class_output = self.class_classifier(x)
        return class_output


class GCNModel(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(GCNModel, self).__init__()
        input_size = num_features + edge_features_size

        # 图卷积层
        self.gcn1 = GCNConv(input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        # 批归一化
        self.norm1 = nn.BatchNorm1d(input_size)

        # 分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x, edge_index, edge_weight=None, edge_features=None, adj=None, T=None):

        x = self.norm1(x)

        # GCN 层
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)

        # 分类
        out = self.class_classifier(x)
        return out
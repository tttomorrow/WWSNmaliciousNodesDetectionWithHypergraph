import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HypergraphConvLayer, GraphConvolution, CustomBatchNorm, HypergraphConvLayerwoL1
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


class HypergraphModelwohypergraph(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(HypergraphModelwohypergraph, self).__init__()
        # 初始化两个超图卷积层
        self.layer1 = GCNConv(num_features, hidden_size)  # 第一个超图卷积层，输出特征为64
        self.layer2 = GCNConv(hidden_size, hidden_size)  # 第二个超图卷积层，输出特征为8
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
        x = self.layer1(x, edge_index)  # 通过第一个超图卷积层
        x = self.layer2(x, edge_index)  # 通过第二个超图卷积层

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


class HypergraphModelOnlyEdgeFeature(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(HypergraphModelOnlyEdgeFeature, self).__init__()
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
        # x = self.norm1(x)
        # # print(x)
        # x = self.layer1(x, edge_index, edge_weight)  # 通过第一个超图卷积层
        # x = self.layer2(x, edge_index, edge_weight)  # 通过第二个超图卷积层
        #
        # x = F.relu(x)  # 激活函数
        #
        # # 聚合节点特征
        # node_features = self.aggregate_node_features(x, edge_index)  # 聚合超图卷积得到的节点特征

        edge_features = self.norm2(edge_features)
        # shared_feature = self.edge_conv(node_features, edge_features, adj, T)  # 将边特征融合到节点特征中

        # shared_feature = self.layer2(shared_feature, edge_index, edge_weight)
        # 分类结果
        class_output = self.class_classifier(edge_features)

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


# 定义整体超图模型
class HypergraphModelwoL1(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(HypergraphModelwoL1, self).__init__()
        # 初始化两个超图卷积层
        self.layer1 = HypergraphConvLayerwoL1(num_features, hidden_size)  # 第一个超图卷积层，输出特征为64
        self.layer2 = HypergraphConvLayerwoL1(hidden_size, hidden_size)  # 第二个超图卷积层，输出特征为8
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
        x = self.layer1(x, edge_index)  # 通过第一个超图卷积层
        x = self.layer2(x, edge_index)  # 通过第二个超图卷积层

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


# 定义整体超图模型
class HypergraphModelwoLinkFusion(nn.Module):
    def __init__(self, num_features, edge_features_size, hidden_size):
        super(HypergraphModelwoLinkFusion, self).__init__()
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

        # edge_features = self.norm2(edge_features)
        # shared_feature = self.edge_conv(node_features, edge_features, adj, T)  # 将边特征融合到节点特征中

        # shared_feature = self.layer2(shared_feature, edge_index, edge_weight)
        # 分类结果
        class_output = self.class_classifier(node_features)
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
        self.gat1 = GATConv(num_features+edge_features_size, hidden_size, heads=2, concat=True, dropout=0.0)
        self.gat2 = GATConv(hidden_size * 2, hidden_size, heads=1, concat=False, dropout=0.1)
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




import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    """
    用于生成“伪样本特征”的生成器（不使用图结构）
    - 输入：随机噪声 (以及可选条件向量)
    - 输出：伪造特征向量 fake_x，维度与真实特征相同
    """
    def __init__(self, num_features, noise_dim=32, hidden_size=128, cond_dim=0):
        """
        num_features : 特征维度（真实样本 x 的维度）
        noise_dim    : 噪声 z 的维度
        hidden_size  : 隐层大小
        cond_dim     : 条件维度（比如 one-hot 类别），没有条件就填 0
        """
        super(FeatureGenerator, self).__init__()
        self.num_features = num_features
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim

        in_dim = noise_dim + cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, num_features),
            # 根据你的数据情况，是否加 Tanh/Sigmoid 自己决定
            # nn.Tanh()
        )

    def forward(self, x, edge_index=None, edge_weight=None, edge_features=None,
                adj=None, T=None, noise=None, cond=None):
        """
        为了兼容 GraphSAGEModel 的调用习惯，保留了 x/edge_index/... 参数，但只用 noise/cond。

        x    : [N, num_features]（这里不使用，只是占位）
        noise: [N, noise_dim]，若为 None 则自动采样
        cond : [N, cond_dim]，可选条件（如类别 one-hot），没有就传 None
        """
        device = x.device if x is not None else (noise.device if noise is not None else "cpu")

        # 推断 batch 大小
        if noise is not None:
            batch_size = noise.size(0)
        elif x is not None:
            batch_size = x.size(0)
        elif cond is not None:
            batch_size = cond.size(0)
        else:
            raise ValueError("至少需要提供 x、noise 或 cond 之一以确定 batch 大小")

        # 若未给 noise，则内部采样
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)

        # 若有条件，则拼接在一起；否则只用噪声
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("构造时设定了 cond_dim>0，但 forward 未传入 cond")
            z = torch.cat([noise, cond], dim=-1)  # [N, noise_dim + cond_dim]
        else:
            z = noise  # [N, noise_dim]

        fake_x = self.net(z)  # [N, num_features]
        return fake_x


class FeatureDiscriminator(nn.Module):
    """
    用于区分“真实特征 / 生成特征”的判别器（不使用图结构）
    - 输入：特征向量 x（可以是真实或生成）
    - 输出：logits（未过 sigmoid）和 prob（过 sigmoid 后的真样本概率）
    """
    def __init__(self, num_features, hidden_size=128, cond_dim=0):
        super(FeatureDiscriminator, self).__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim

        in_dim = num_features + cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_size, 1)  # 输出一个 logit
        )

    def forward(self, x, edge_index=None, edge_weight=None, edge_features=None,
                adj=None, T=None, cond=None):
        """
        x   : [N, num_features]（真实或生成）
        cond: [N, cond_dim] 条件（如类别 one-hot），若无需条件则为 None
        其余图相关参数仅为兼容接口，不使用
        """
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("构造时设定了 cond_dim>0，但 forward 未传入 cond")
            x_in = torch.cat([x, cond], dim=-1)  # [N, num_features + cond_dim]
        else:
            x_in = x  # [N, num_features]

        logits = self.net(x_in).view(-1)  # [N]
        prob = torch.sigmoid(logits)
        return logits, prob


class FeatureGAN(nn.Module):
    """
    一个简单封装，把生成器和判别器放在一起，方便管理。
    """
    def __init__(self, num_features, noise_dim=32, hidden_size=128, cond_dim=0):
        super(FeatureGAN, self).__init__()
        self.generator = FeatureGenerator(
            num_features=num_features,
            noise_dim=noise_dim,
            hidden_size=hidden_size,
            cond_dim=cond_dim
        )
        self.discriminator = FeatureDiscriminator(
            num_features=num_features,
            hidden_size=hidden_size,
            cond_dim=cond_dim
        )

    def generate(self, x, edge_index=None, edge_weight=None, edge_features=None,
                 adj=None, T=None, noise=None, cond=None):
        """
        只调用生成器，返回 fake_x
        """
        fake_x = self.generator(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_features=edge_features,
            adj=adj,
            T=T,
            noise=noise,
            cond=cond
        )
        return fake_x

    def discriminate(self, x, edge_index=None, edge_weight=None, edge_features=None,
                     adj=None, T=None, cond=None):
        """
        只调用判别器，返回 logits, prob
        """
        logits, prob = self.discriminator(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_features=edge_features,
            adj=adj,
            T=T,
            cond=cond
        )
        return logits, prob

import pandas as pd
import numpy as np
import re
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


def parse_log_file(log_file, time_interval):
    """
    解析日志文件，提取恶意节点、黑洞节点、丢包信息，并按时间段整理。
    返回字典，包含每个时间段内的丢包信息以及恶意和黑洞节点。
    """
    malicious_nodes = set()
    blackhole_nodes = set()
    drop_packet_info = {}

    # 解析日志文件
    with open(log_file, 'r') as file:
        for line in file:
            # 解析黑洞节点
            blackhole_match = re.match(r"Node (\d+) marked for Blackhole.", line)
            if blackhole_match:
                node_id = int(blackhole_match.group(1))
                blackhole_nodes.add(node_id)

            # 解析恶意节点（根据实际情况调整匹配规则）
            malicious_match = re.match(r"randomNumbers(\d+)", line)
            if malicious_match:
                node_id = int(malicious_match.group(1))
                malicious_nodes.add(node_id)

            # 解析丢包信息
            drop_match = re.match(r"\[node (\d+)] time:(\d+\.\d+), Drop packet (\d+)", line)
            if drop_match:
                node_id = int(drop_match.group(1))
                timestamp = float(drop_match.group(2))

                # 根据时间段分组丢包信息
                time_segment = np.floor(timestamp / time_interval) * time_interval

                if time_segment not in drop_packet_info:
                    drop_packet_info[time_segment] = {}

                if node_id not in drop_packet_info[time_segment]:
                    drop_packet_info[time_segment][node_id] = 0

                drop_packet_info[time_segment][node_id] += 1

    return {
        'malicious_nodes': malicious_nodes,
        'blackhole_nodes': blackhole_nodes,
        'drop_packet_info': drop_packet_info
    }


def process_csv(input_file, output_file, log_file, time_interval=2.50):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    # print(df.columns)  # 查看列名

    # 解析日志文件
    log_data = parse_log_file(log_file, time_interval)

    # 确保时间列是浮动类型
    df['Time'] = df['Time'].astype(float)

    # 创建时间段列（按时间间隔分段）
    df['TimeSegment'] = np.floor(df['Time'] / time_interval) * time_interval

    # 初始化结果列表
    result = []

    # 按时间段分组，遍历每个时间段
    for time_segment, group in df.groupby('TimeSegment'):
        # 按照监听节点和被监听节点分组
        for listener, sub_group in group.groupby('listener'):
            for src_node_id, src_group in sub_group.groupby('SrcNodeId'):
                # 计算信道质量指标（例如：SNR, SignalPower, NoisePower的均值）
                avg_snr = src_group['SNR'].mean()
                avg_signal_power = src_group['SignalPower'].mean()
                avg_noise_power = src_group['NoisePower'].mean()

                # 统计每个数据包类型的发送次数
                packet_counts = src_group['PacketType'].value_counts().to_dict()

                # 获取UDP、ARP、ADOV的发送次数，其他包类型不统计
                udp_count = packet_counts.get('UDP', 0)
                arp_request_count = packet_counts.get('ARP Request', 0)
                arp_reply_count = packet_counts.get('ARP Replay', 0)
                route_request_count = packet_counts.get('Route Request', 0)
                route_reply_count = packet_counts.get('Route Replay', 0)
                route_error_count = packet_counts.get('Route Error', 0)
                route_reply_ack_count = packet_counts.get('Route Replay ACK', 0)
                ack_count = packet_counts.get('ACK', 0)

                # 获取当前时间段丢包次数
                drop_count = log_data['drop_packet_info'].get(time_segment, {}).get(src_node_id, 0)

                # 是否是恶意节点、黑洞节点
                is_malicious = 1 if src_node_id in log_data['malicious_nodes'] else 0
                is_blackhole = 1 if src_node_id in log_data['blackhole_nodes'] else 0
                is_selective_forwarding = 0  # 暂时没有选择性转发节点的具体信息

                # 创建结果记录
                result.append({
                    'ListenerNode': listener,
                    'SrcNodeId': src_node_id - 1,
                    'UDPCount': udp_count,
                    'ARPRequestCount': arp_request_count,
                    'ARPReplayCount': arp_reply_count,
                    'RouteRequestCount': route_request_count,
                    'RouteReplayCount': route_reply_count,
                    'RouteErrorCount': route_error_count,
                    'RouteReplayACKCount': route_reply_ack_count,
                    'ACKCount': ack_count,
                    'AvgSNR': avg_snr,
                    'AvgSignalPower': avg_signal_power,
                    'AvgNoisePower': avg_noise_power,
                    'PacketLossCount': drop_count,
                    'IsMaliciousNode': is_malicious,
                    'IsBlackholeNode': is_blackhole,
                    'IsSelectiveForwardingNode': is_selective_forwarding,
                    'TimeSegment': time_segment,
                    'TimeID': int(time_segment / time_interval)
                })

    # 将结果转换为 DataFrame
    result_df = pd.DataFrame(result)

    # 保存结果到 CSV 文件
    result_df.to_csv(output_file, index=False)


def process_ns3_data(df):
    # 在时间段内按源节点ID排序
    df = df.sort_values(by="SrcNodeId")
    # 步骤1：提取节点的特征
    # 第一组特征：UDPCount, ARPRequestCount, ARPReplayCount, RouteRequestCount, RouteReplayCount, RouteErrorCount, RouteReplayACKCount, ACKCount
    node_features_group_1 = df[['UDPCount', 'ARPRequestCount', 'ARPReplayCount',
                                'RouteRequestCount', 'RouteReplayCount', 'RouteErrorCount',
                                'RouteReplayACKCount', 'ACKCount']].values
    node_features_group_1 = torch.tensor(node_features_group_1, dtype=torch.int)
    num_feature1 = node_features_group_1.size(1)

    # 第二组特征：AvgSNR, AvgSignalPower, AvgNoisePower
    node_features_group_2 = df[['AvgSNR', 'AvgSignalPower', 'AvgNoisePower']].values
    node_features_group_2 = torch.tensor(node_features_group_2, dtype=torch.float)

    # 合并这两组特征
    node_features = torch.cat([node_features_group_1, node_features_group_2], dim=1)

    # 步骤2：构建有向图结构：用一个字典来表示图，键为节点，值为邻接节点
    graph = {}  # 用于存储图的邻接信息

    for _, row in df.iterrows():
        listener_node = row['ListenerNode']  # 监听节点
        src_node = row['SrcNodeId']  # 被监听节点

        # 将被监听节点和监听节点之间建立边的关系
        if src_node not in graph:
            graph[src_node] = []
        if listener_node not in graph:
            graph[listener_node] = []

        # 双向边关系（由于假设每个监听节点和被监听节点互为邻接）
        graph[src_node].append(listener_node)

    total_edges = sum(len(neighbors) for neighbors in graph.values())
    # print(f"Total number of edges (directed graph): {total_edges}")
    # 使用 networkx 创建有向图
    G = nx.DiGraph()  # 使用 DiGraph 来创建有向图

    # 将邻接表字典添加到图中
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # 获取邻接矩阵（有向图）
    adj = nx.adjacency_matrix(G)

    # 步骤3：构建超边（每个目标节点（被监听节点）对应一个超边，包含所有与该节点相关的源节点）
    hyperedges = {}  # 用字典存储每个目标节点对应的所有源节点（超边）

    # 步骤4：为每条边分配唯一的ID
    edge_map = {}  # 用于存储边的唯一ID
    edge_id_counter = 0  # 边ID计数器
    # num_edges = len(df)  # 边的数量
    # edge_adjacency = np.zeros((num_edges, num_edges))  # 初始化邻接矩阵

    for _, row in df.iterrows():
        listener_node = row['ListenerNode']  # 监听节点
        src_node = row['SrcNodeId']  # 被监听节点
        edge = (src_node, listener_node)  # 边由被监听节点和监听节点组成

        if edge not in edge_map:
            edge_map[edge] = edge_id_counter  # 为该边分配唯一ID
            edge_id_counter += 1

        # 将监听节点加入到对应的被监听节点的超边中
        if src_node not in hyperedges:
            hyperedges[src_node] = []
        hyperedges[src_node].append(edge_map[edge])  # 将边的ID加入超边中

    # print('adj.shape')
    # print(adj.shape)
    # create transform matrix T, dimension is num_node * num_edge in the graph
    T = create_transition_matrix(adj)
    # print('T.shape')
    # print(T.shape)
    T = sparse_mx_to_torch_sparse_tensor(T)
    # print('T.shape')
    # print(T.shape)
    # create edge adjacent matrix from node/vertex adjacent matrix
    eadj, edge_name = create_edge_adj(adj)
    # print('eadj.shape')
    # print(eadj.shape)
    eadj = sparse_mx_to_torch_sparse_tensor(normalize(eadj))

    # 步骤5：构建超图中的边和节点特征
    edge_index = []  # 存储超边中的节点连接关系
    hypernode_features = []  # 存储每个节点的特征
    hyper_edge_ids = []  # 存储每条边的ID（即超图节点ID）

    hyper_edge_id = 0
    for src_node, edges in hyperedges.items():
        # 每个源节点（被监听节点）对应一个超边
        for ori_edge_id in edges:
            edge_index.append([hyper_edge_id, ori_edge_id])  # 每条边连接同一个原始边ID的节点
            # 将该边对应的特征（第二组特征）添加到超图节点特征中

            hyper_node_feature = node_features[ori_edge_id]  # 监听节点的特征，此处包含链路特征
            hypernode_features.append(hyper_node_feature)  # 将监听节点的特征添加到超图节点特征中
            hyper_edge_ids.append(hyper_edge_id)  # 记录边的ID作为超图节点ID
        hyper_edge_id = hyper_edge_id + 1

    # 步骤6：计算每个超边的相似度矩阵
    edge_weights = []  # 用于存储每个超边的相似度矩阵

    data_edge_index = np.array(edge_index)  # 将 edge_index 转化为 NumPy 数组

    grouped = {}  # 提取第一个值相同的第二个值

    # 遍历数据，按第一个值分组
    for pair in data_edge_index:
        key = pair[0]
        value = pair[1]
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(value)

    grouped_result = []  # 创建二维数组，保存每个组的第二个值

    for key in sorted(grouped.keys()):  # 将每个组的第二个值作为一维数组添加到结果中
        grouped_result.append(grouped[key])

    # 将结果转换为 NumPy 数组，最终是一个二维数组
    grouped_result = np.array(grouped_result, dtype=object)  # 使用 dtype=object 来支持不等长的列表
    print(hypernode_features)
    np_hypernode_features = np.array(hypernode_features)  # 转换为 NumPy 数组
    print(np_hypernode_features.shape)
    for i in range(grouped_result.size):
        index = np.array(grouped_result[i - 1])
        np_hypernode_behavior_features = np_hypernode_features[index][0: num_feature1 - 1]
        print(np_hypernode_behavior_features)
        similarity_matrix = cosine_similarity(np_hypernode_behavior_features,
                                              np_hypernode_behavior_features)
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float)
        row_sum = similarity_matrix.sum(dim=0, keepdim=True)
        edge_weights.append(row_sum / row_sum.sum(dim=1))

    # 计算每个张量的均值，并将其存储为边权重
    hyperedge_weights = [tensor.mean().item() for tensor in edge_weights]

    # 步骤7：将边和节点特征转化为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = edge_index[[1, 0]]
    hypernode_features = torch.stack(hypernode_features)  # 将节点特征堆叠成一个矩阵
    edge_ids = torch.tensor(hyper_edge_ids, dtype=torch.long)  # 边的ID（原始边ID）
    hyperedge_weights = torch.tensor(hyperedge_weights)
    hyperedge_weights = torch.nan_to_num(hyperedge_weights, nan=0.0)  # 使用 torch.nan_to_num() 将 NaN 替换为 0

    # 步骤8：标签提取（IsMaliciousNode）
    df_unique = df.drop_duplicates(subset=['SrcNodeId'], keep='first')

    # 按照 'SrcNodeId' 排序
    df_sorted = df_unique.sort_values(by="SrcNodeId")

    # 提取标签列 'IsMaliciousNode'
    labels = df_sorted['IsMaliciousNode'].values

    # 转换为 PyTorch tensor 格式
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 步骤10：返回PyG数据对象
    data = Data(x=hypernode_features, edge_index=edge_index, y=labels_tensor,
                edge_ids=edge_ids, edge_weights=hyperedge_weights)  # 添加边ID作为超图节点ID，返回的x包含链路特征，特征最后三列

    adj = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0])))

    return data, adj, T


def process_ns3_data_by_timeid(df):
    # 根据 TimeID 分组并处理每个时间段的数据
    timeid_groups = df.groupby('TimeID')

    # 为每个 TimeID 创建一个独立的超图数据集
    data_per_timeid = []
    adj_per_timeid = []
    T_per_timeid = []

    for timeid, group in timeid_groups:
        data, adj, T = process_ns3_data(group)
        data_per_timeid.append(data)
        adj_per_timeid.append(adj)
        T_per_timeid.append(T)

    return data_per_timeid, adj_per_timeid, T_per_timeid, len(timeid_groups)


# def process_wwsn_data(df):


def create_edge_adj(vertex_adj):

    # 获取邻接矩阵中所有非零元素的索引，表示图中存在的边
    edge_index = np.nonzero(vertex_adj)

    # 计算边的数量
    num_edge = len(edge_index[0])

    # 创建一个边名列表，每个元素是一个二元组 (src_node, tgt_node)，表示有向边
    edge_name = [(edge_index[0][i], edge_index[1][i]) for i in range(num_edge)]

    # 初始化一个大小为 num_edge x num_edge 的边邻接矩阵
    edge_adj = np.zeros((num_edge, num_edge))

    # 遍历所有边对，检查是否存在共同的节点（无论是源节点还是目标节点）
    for i in range(num_edge):
        for j in range(i, num_edge):
            # 如果边 i 和 边 j 有共同的节点，则这两条边是相连的
            if len(set(edge_name[i]) & set(edge_name[j])) > 0:
                edge_adj[i, j] = 1

    # 返回的邻接矩阵需要是稀疏矩阵，使用 csr_matrix 来节省内存
    adj = sp.csr_matrix(edge_adj)

    # 返回边邻接矩阵和边列表
    return adj, edge_name


def create_transition_matrix(vertex_adj):
    '''create N_v * N_e transition matrix'''
    # 不去除自环，确保所有边都被计入
    # vertex_adj.setdiag(0)  # 这行可以注释掉，如果不想移除自环

    # 获取邻接矩阵中所有的非零元素，即边的索引
    edge_index = np.nonzero(vertex_adj)  # 获取所有边的索引
    num_edge = len(edge_index[0])  # 边的数量

    # 输出边的数量
    # print('Number of edges:', num_edge)

    # 创建一个二值矩阵 T，大小是节点数 x 边数
    # T[i, m] = 1 如果节点 i 参与边 m，0 否则
    T = np.zeros((vertex_adj.shape[0], num_edge), dtype=int)

    # 填充 T 矩阵
    for m, (src, tgt) in enumerate(zip(edge_index[0], edge_index[1])):
        T[src, m] = 1  # 源节点连接到边 m
        T[tgt, m] = 1  # 目标节点也连接到边 m（如果需要）

    # 将 T 转换为稀疏矩阵格式（如果需要节省内存）
    T_sparse = sp.csr_matrix(T)

    return T_sparse


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(data_type, data_file):
    if data_type == 1:  # ns3仿真数据
        df = pd.read_csv(data_file)
        processed_data, adj, T, group_num = process_ns3_data_by_timeid(df[0:1000])
    # elif data_type == 0:  # wwsn实验数据
    #     df = pd.read_csv(data_file)
    #     processed_data = process_wwsn_data(df)

    return processed_data, adj, T, group_num

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
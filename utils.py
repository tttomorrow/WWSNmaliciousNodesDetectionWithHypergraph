import pandas as pd
import numpy as np
import re
import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict


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


def process_ns3_csv(input_rx_file, output_file, log_file, time_interval=10.0):
    # 读取 RX CSV 文件
    df = pd.read_csv(input_rx_file)

    # print(df.columns)  # 查看列名

    # 解析日志文件
    log_data = parse_log_file(log_file, time_interval)

    # 确保时间列是浮动类型
    df['Time'] = df['Time'].astype(float)

    # 创建时间段列（按时间间隔分段）
    df['TimeSegment'] = np.floor(df['Time'] / time_interval) * time_interval

    # 初始化结果列表
    result = []
    is_sink = 0

    # 按时间段分组，遍历每个时间段
    for time_segment, group in df.groupby('TimeSegment'):
        # # 先统计接收数据包
        rec_udp_count = defaultdict(int)
        rec_arp_request_count = defaultdict(int)
        rec_arp_reply_count = defaultdict(int)
        rec_route_request_count = defaultdict(int)
        rec_route_reply_count = defaultdict(int)
        rec_route_error_count = defaultdict(int)
        rec_route_reply_ack_count = defaultdict(int)
        rec_ack_count = defaultdict(int)
        for des_tx, sub_group_tx in group.groupby('RecNodeId'):
            if des_tx == 255:
                continue
            # 注意！！！！！！！！！！！！！！！！！！！！！！          RX文件中sourceID比正确值大1
            # 统计每个数据包类型的发送次数, TX文件中sourceID比正确值大1， listeneriD则比正确值小1
            # Tx中的source_tx为数据包的目的节点
            rec_packets_count = sub_group_tx['PacketType'].value_counts().to_dict()
            # 获取UDP、ARP、ADOV的发送次数，其他包类型不统计
            rec_udp_count[des_tx] = rec_packets_count.get('UDP', 0)
            rec_arp_request_count[des_tx] = rec_packets_count.get('ARP Request', 0)
            rec_arp_reply_count[des_tx] = rec_packets_count.get('ARP Replay', 0)
            rec_route_request_count[des_tx] = rec_packets_count.get('Route Request', 0)
            rec_route_reply_count[des_tx] = rec_packets_count.get('Route Replay', 0)
            rec_route_error_count[des_tx] = rec_packets_count.get('Route Error', 0)
            rec_route_reply_ack_count[des_tx] = rec_packets_count.get('Route Replay ACK', 0)
            rec_ack_count[des_tx] = rec_packets_count.get('ACK', 0)
        # print(rec_udp_count)

        # # 再统计转发数据包
        fw_udp_count = defaultdict(int)
        fw_arp_request_count = defaultdict(int)
        fw_arp_reply_count = defaultdict(int)
        fw_route_request_count = defaultdict(int)
        fw_route_reply_count = defaultdict(int)
        fw_route_error_count = defaultdict(int)
        fw_route_reply_ack_count = defaultdict(int)
        fw_ack_count = defaultdict(int)
        for des_tx, sub_group_tx in group.groupby('FwNodeId'):
            if des_tx == 255:
                continue
            # 注意！！！！！！！！！！！！！！！！！！！！！！          RX文件中sourceID比正确值大1
            # 统计每个数据包类型的发送次数, TX文件中sourceID比正确值大1， listeneriD则比正确值小1
            # Tx中的source_tx为数据包的目的节点
            fw_packets_count = sub_group_tx['PacketType'].value_counts().to_dict()
            # 获取UDP、ARP、ADOV的发送次数，其他包类型不统计
            fw_udp_count[des_tx] = fw_packets_count.get('UDP', 0)
            fw_arp_request_count[des_tx] = fw_packets_count.get('ARP Request', 0)
            fw_arp_reply_count[des_tx] = fw_packets_count.get('ARP Replay', 0)
            fw_route_request_count[des_tx] = fw_packets_count.get('Route Request', 0)
            fw_route_reply_count[des_tx] = fw_packets_count.get('Route Replay', 0)
            fw_route_error_count[des_tx] = fw_packets_count.get('Route Error', 0)
            fw_route_reply_ack_count[des_tx] = fw_packets_count.get('Route Replay ACK', 0)
            fw_ack_count[des_tx] = fw_packets_count.get('ACK', 0)

        # 按照监听节点和被监听节点分组
        for listener, sub_group in group.groupby('listener'):
            for src_node_id, src_group in sub_group.groupby('FwNodeId'):
                if src_node_id == 0:
                    continue
                is_sink = 0  # 标记sink节点
                # 计算信道质量指标（例如：SNR, SignalPower, NoisePower的均值）
                avg_snr = src_group['SNR'].mean()
                avg_signal_power = src_group['SignalPower'].mean()
                avg_noise_power = src_group['NoisePower'].mean()

                # 统计每个数据包类型的接收次数
                packet_counts = src_group['PacketType'].value_counts().to_dict()

                # 获取UDP、ARP、ADOV的接收次数，其他包类型不统计
                udp_count = packet_counts.get('UDP', 0)
                arp_request_count = packet_counts.get('ARP Request', 0)
                arp_reply_count = packet_counts.get('ARP Replay', 0)
                route_request_count = packet_counts.get('Route Request', 0)
                route_reply_count = packet_counts.get('Route Replay', 0)
                route_error_count = packet_counts.get('Route Error', 0)
                route_reply_ack_count = packet_counts.get('Route Replay ACK', 0)
                ack_count = packet_counts.get('ACK', 0)

                # 获取当前时间段丢包次数
                # ！！！！！！！！！！！！！！！！ns3中节点id从0开始计算，本代码所有id从1开始，故此处加1
                drop_count = log_data['drop_packet_info'].get(time_segment, {}).get(src_node_id - 1, 0)

                # 是否是恶意节点、黑洞节点
                # ！！！！！！！！！！！！！！！！ns3中节点id从0开始计算，本代码所有id从1开始
                is_malicious = 1 if src_node_id - 1 in log_data['malicious_nodes'] else 0
                is_blackhole = 1 if src_node_id - 1 in log_data['blackhole_nodes'] else 0
                if src_node_id == 1:
                    is_sink = 1
                is_selective_forwarding = 0  # 暂时没有选择性转发节点的具体信息
                result.append({
                    'ListenerNode': listener,
                    'SrcNodeId': src_node_id,
                    'SendUDPCount': udp_count,
                    'SendARPRequestCount': arp_request_count,
                    'SendARPReplayCount': arp_reply_count,
                    'SendRouteRequestCount': route_request_count,
                    'SendRouteReplayCount': route_reply_count,
                    'SendRouteErrorCount': route_error_count,
                    'SendRouteReplayACKCount': route_reply_ack_count,
                    'FwUDPCount': fw_udp_count[src_node_id] - udp_count,
                    'FwARPRequestCount': fw_arp_request_count[src_node_id] - arp_request_count,
                    'FwARPReplayCount': fw_arp_reply_count[src_node_id] - arp_reply_count,
                    'FwRouteRequestCount': fw_route_request_count[src_node_id] - route_request_count,
                    'FwRouteReplayCount': fw_route_reply_count[src_node_id] - route_reply_count,
                    'FwRouteErrorCount': fw_route_error_count[src_node_id] - route_error_count,
                    'FwRouteReplayACKCount': fw_route_reply_ack_count[src_node_id] - route_reply_ack_count,
                    'RecUDPCount': rec_udp_count[src_node_id],
                    'RecARPRequestCount': rec_arp_request_count[src_node_id],
                    'RecARPReplayCount': rec_arp_reply_count[src_node_id],
                    'RecRouteRequestCount': rec_route_request_count[src_node_id],
                    'RecRouteReplayCount': rec_route_reply_count[src_node_id],
                    'RecRouteErrorCount': rec_route_error_count[src_node_id],
                    'RecRouteReplayACKCount': rec_route_reply_ack_count[src_node_id],
                    'RecACKCount': rec_ack_count[src_node_id],
                    'IsSink': is_sink,
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

        # 在每个时间段内，处理反向边逻辑
        # 获取 ListenerNode 和 SrcNodeId 的集合
        listener_nodes = set(group['listener'])
        src_nodes = set(group['SrcNodeId'])

        # 找到不在 SrcNodeId 中的 ListenerNode
        missing_listener_nodes = listener_nodes - src_nodes

        if missing_listener_nodes:
            print(missing_listener_nodes)
            # 新的数据列表（反向边）
            new_edges = []

            # 为每个缺失的 ListenerNode 添加反向边
            for missing_listener in missing_listener_nodes:
                is_sink = 0  # 标记sink节点
                # 找到该 ListenerNode 的所有出边（即原数据中的 ListenerNode 为 missing_listener 的边）
                original_edges = group[group['listener'] == missing_listener]
                # 去除重复的边
                original_edges = original_edges.drop_duplicates(subset=['listener'])
                for _, row in original_edges.iterrows():
                    # 反向边的 ListenerNode 和 SrcNodeId 对调，特征值全部为 0
                    # ！！！！！！！！！！！！！！！！ns3中节点id从0开始计算，本代码所有id从1开始，故此处减1
                    is_malicious = 1 if row['listener'] - 1 in log_data['malicious_nodes'] else 0
                    is_blackhole = 1 if row['listener'] - 1 in log_data['blackhole_nodes'] else 0

                    # 获取当前时间段丢包次数
                    # ！！！！！！！！！！！！！！！！ns3中节点id从0开始计算，本代码所有id从1开始，故此处加1
                    drop_count = log_data['drop_packet_info'].get(time_segment, {}).get(row['listener'] - 1, 0)
                    if row['listener'] == 1:
                        is_sink = 1
                    new_edge = {
                        'ListenerNode': row['SrcNodeId'],
                        'SrcNodeId': row['listener'],
                        'SendUDPCount': 0,
                        'SendARPRequestCount': 0,
                        'SendARPReplayCount': 0,
                        'SendRouteRequestCount': 0,
                        'SendRouteReplayCount': 0,
                        'SendRouteErrorCount': 0,
                        'SendRouteReplayACKCount': 0,
                        'FwUDPCount': 0,
                        'FwARPRequestCount': 0,
                        'FwARPReplayCount': 0,
                        'FwRouteRequestCount': 0,
                        'FwRouteReplayCount': 0,
                        'FwRouteErrorCount': 0,
                        'FwRouteReplayACKCount': 0,
                        'RecUDPCount': 0,
                        'RecARPRequestCount': 0,
                        'RecARPReplayCount': 0,
                        'RecRouteRequestCount': 0,
                        'RecRouteReplayCount': 0,
                        'RecRouteErrorCount': 0,
                        'RecRouteReplayACKCount': 0,
                        'RecACKCount': 0,
                        'IsSink': is_sink,
                        'AvgSNR': 0,
                        'AvgSignalPower': 0,
                        'AvgNoisePower': 0,
                        'PacketLossCount': drop_count,
                        'IsMaliciousNode': is_malicious,
                        'IsBlackholeNode': is_blackhole,
                        'IsSelectiveForwardingNode': 0,  # 仍然没有选择性转发节点的具体信息
                        'TimeSegment': time_segment,
                        'TimeID': int(time_segment / time_interval)
                    }
                    result.append(new_edge)
        else:
            continue

    # 将结果转换为 DataFrame
    result_df = pd.DataFrame(result)

    # 保存结果到 CSV 文件
    result_df.to_csv(output_file, index=False)


def process_ns3_data(df):
    # 在时间段内按源节点ID排序
    df = df.sort_values(by="SrcNodeId")
    # 步骤1：提取节点的特征 第一组特征：UDPCount, ARPRequestCount, ARPReplayCount, RouteRequestCount, RouteReplayCount,
    # RouteErrorCount, RouteReplayACKCount, ACKCount
    node_features_group_1 = df[['SendUDPCount',
                                'SendARPRequestCount',
                                'SendARPReplayCount',
                                'SendRouteRequestCount',
                                'SendRouteReplayCount',
                                'SendRouteErrorCount',
                                'SendRouteReplayACKCount',
                                'FwUDPCount',
                                'FwARPRequestCount',
                                'FwARPReplayCount',
                                'FwRouteRequestCount',
                                'FwRouteReplayCount',
                                'FwRouteErrorCount',
                                'FwRouteReplayACKCount',
                                'RecUDPCount',
                                'RecARPRequestCount',
                                'RecARPReplayCount',
                                'RecRouteRequestCount',
                                'RecRouteReplayCount',
                                'RecRouteErrorCount',
                                'RecRouteReplayACKCount',
                                'RecACKCount',
                                'IsSink']].values
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

    # total_edges = sum(len(neighbors) for neighbors in graph.values())
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
    np_hypernode_features = np.array(hypernode_features)  # 转换为 NumPy 数组

    for i in range(grouped_result.size):
        index = np.array(grouped_result[i])
        # print(np_hypernode_features[index, 0: num_feature1 - 1].shape)
        similarity_matrix = cosine_similarity(np_hypernode_features[index, 0: num_feature1 - 1],
                                              np_hypernode_features[index, 0: num_feature1 - 1])
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

    # # 按照 'SrcNodeId' 排序
    # df_sorted = df_unique.sort_values(by="SrcNodeId")

    # 提取标签列 'IsMaliciousNode'
    labels = df_unique['IsMaliciousNode'].values

    # 进行独热编码，创建一个大小为 (n_samples, n_classes) 的矩阵
    # 这里n_classes是2，因为有0和1两个类别
    labels_one_hot = F.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=2)

    # labels_tensor 现在是一个二维张量，每一行是独热编码表示
    labels_tensor = labels_one_hot.float()  # 转换为float类型（常用于分类任务）

    # 步骤10：返回PyG数据对象
    data = Data(x=hypernode_features[:, 0:-3], edge_attr=hypernode_features[:, -3:], edge_index=edge_index,
                y=labels_tensor,
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

    return data_per_timeid, adj_per_timeid, T_per_timeid, len(timeid_groups) - 1


# def process_wwsn_data(df):


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
        processed_data, adj, T, group_num = process_ns3_data_by_timeid(df)
    # elif data_type == 0:  # wwsn实验数据
    #     df = pd.read_csv(data_file)
    #     processed_data = process_wwsn_data(df)

    return processed_data, adj, T, group_num


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels[:, 1]).double()
    correct = correct.sum()
    return correct / len(labels)


def auc(output, labels):
    # 假设 output 是模型的 raw 输出（logits），labels 是 ground truth 的标签
    # 这里假设输出是二维张量，其中第二列是正类的概率（例如，经过 softmax 或 sigmoid）

    # 如果模型输出的是 logits，通常需要进行 softmax 或 sigmoid 转换
    probs = torch.nn.functional.softmax(output, dim=1)[:, 1]  # 对于二分类问题，取正类的概率

    # 确保 labels 是一维的，并检查标签中的类别数量
    if len(np.unique(labels[:, 1].cpu().numpy())) == 2:
        # 计算 AUC 分数
        auc_score = roc_auc_score(labels[:, 1].cpu().numpy(), probs.cpu().detach().numpy())
        # print(auc_score)
    else:
        # print("Warning: Only one class present in labels.")
        auc_score = 0  # 或者可以返回其他默认值或提示
    return auc_score


def precision(output, labels):
    preds = output.max(1)[1]  # 获取预测的类别标签
    tp = ((preds == 1) & (labels[:, 1] == 1)).sum().item()  # 真正类
    fp = ((preds == 1) & (labels[:, 1] == 0)).sum().item()  # 假正类
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # 防止除零错误
    return precision


def recall(output, labels):
    preds = output.max(1)[1]  # 获取预测的类别标签
    tp = ((preds == 1) & (labels[:, 1] == 1)).sum().item()  # 真正类
    fn = ((preds == 0) & (labels[:, 1] == 1)).sum().item()  # 假负类
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # 防止除零错误
    return recall


def f1_score(output, labels):
    prec = precision(output, labels)
    rec = recall(output, labels)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0  # 防止除零错误
    return f1


def confusion_matrix(output, labels):
    preds = output.max(1)[1]  # 获取预测的类别标签
    tp = ((preds == 1) & (labels[:, 1] == 1)).sum().item()  # 真正类
    tn = ((preds == 0) & (labels[:, 1] == 0)).sum().item()  # 真负类
    fp = ((preds == 1) & (labels[:, 1] == 0)).sum().item()  # 假正类
    fn = ((preds == 0) & (labels[:, 1] == 1)).sum().item()  # 假负类

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

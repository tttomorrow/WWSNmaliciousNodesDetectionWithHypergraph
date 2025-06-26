from collections import defaultdict
import networkx as nx
import gpustat
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import torch.nn.functional as F
import pandas as pd
from get_data import load_network
import torch.nn.utils.rnn as rnn_utils
from sklearn.preprocessing import MinMaxScaler

def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count())))  # list(set(X)) is done to shuffle the array
    print(gpus)
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    print(str(gpus[np.argmin(mem)]))
    return str(gpus[np.argmin(mem)])


def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count += 1


def group_interactions_by_item(item_sequence_id):
    item_to_interactions = defaultdict(list)
    for j, item_id in enumerate(item_sequence_id):
        item_to_interactions[item_id].append(j)
    return item_to_interactions

# Define a function to create t-batches based on grouped interactions
def create_t_batches(timestamp_sequence, interaction_groups, tbatch_timespan):
    t_batches = []
    current_t_batch = []

    for item_id, interactions in interaction_groups.items():
        interactions.sort(key=lambda j: timestamp_sequence[j])
        for j in interactions:
            timestamp = timestamp_sequence[j]

            if not current_t_batch or timestamp - current_t_batch[0]["start_time"] <= tbatch_timespan:
                current_t_batch.append({"j": j, "start_time": timestamp})
            else:
                t_batches.append(current_t_batch)
                current_t_batch = [{"j": j, "start_time": timestamp}]

    if current_t_batch:
        t_batches.append(current_t_batch)

    return t_batches




# 构建超图
def construct_hypergraph(user_sequence_id, item_sequence_id, t_batch):
    hypergraph = {"t_batch_users": [], "Item-to-Users Mapping": {}}

    for entry in t_batch:
        user_id = user_sequence_id[entry["j"]]
        item_id = item_sequence_id[entry["j"]]

        # 将用户添加到当前 t-batch 的超图
        hypergraph["t_batch_users"].append(user_id)

        # 将用户与项目关联
        hypergraph["Item-to-Users Mapping"].setdefault(item_id, set()).add(user_id)

    return hypergraph



def process_ns3_data(df):
    # 在时间段内按源节点ID排序
    df = df.sort_values(by="SrcNodeId")
    # 计算三个特征
    node_features_group_1 = df[['PacketLossRate',
                        'DataReportRate',
                        'DataForwardRate']].values

    # print(new_node_features_group_1)
    new_node_features_group_1 = torch.tensor(node_features_group_1, dtype=torch.float)
    num_feature1 = new_node_features_group_1.size(1)



    # 第二组特征：AvgSNR, AvgSignalPower, AvgNoisePower
    node_features_group_2 = df[['AvgSNR', 'AvgSignalPower', 'AvgNoisePower']].values
    node_features_group_2 = torch.tensor(node_features_group_2, dtype=torch.float)

    # 合并这两组特征
    node_features = torch.cat([new_node_features_group_1, node_features_group_2], dim=1)

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

    return data, adj


def process_data_by_timeid(dataset, args):
    if args.network == 1:
        data_name = "wikipedia"
    if args.network == 2:
        data_name = "mooc"
    if args.network == 3:
        data_name = "reddit"
    if args.network == 4:
        data_name = "eucore"
    if args.network == 5:
        data_name = "uci"
    if args.network == 6:
        data_name = "digg"

    if dataset == 'train':
        if args.noise != 0 :
            datapath = "../dataset/" + data_name + "/" + data_name + "_n_" + str(args.noise) + '0' + str(
            int(0.1 * 100)) + '.csv'
        # elif args.anomaly != 0:
        #     datapath = "../dataset/" + data_name + "/" + data_name + "_n_" + str(args.anomaly) + '.csv'
    else:
        # 根据 TimeID 分组并处理每个时间段的数据
        if args.anomaly != 0 and args.noise != 0:
            datapath = "../dataset/" + data_name + "/" + data_name + "_n_" + str(args.noise) + '0' + str(
                int(args.anomaly * 100)) + '.csv'
        elif args.noise != 0:
            datapath = "../dataset/" + data_name + "/" + data_name + "_n_" + str(args.noise) + '.csv'
        else:
            datapath = "../dataset/" + data_name + "/" + data_name + '.csv'

    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence,
     timestamp_sequence, feature_sequence, y_true] = load_network(datapath)

    num_interactions = len(user_sequence_id)
    num_users = len(user2id)
    num_items = len(item2id) + 1  # one extra item for "none-of-these"
    num_features = len(feature_sequence[0])
    true_labels_ratio = len(y_true) / (1.0 + sum(y_true))
    # +1 in denominator in case there are no state change labels, which will throw an error.
    print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (
        num_users, num_items, num_interactions, sum(y_true), len(y_true)))

    # 转为 numpy 数组
    user_sequence_id = np.array(user_sequence_id, dtype=np.int64).reshape(-1, 1)
    item_sequence_id = np.array(item_sequence_id, dtype=np.int64).reshape(-1, 1)
    timestamp_sequence = np.array(timestamp_sequence).reshape(-1, 1)
    feature_sequence = np.array(feature_sequence)  # shape (N, D)
    y_true = np.array(y_true, dtype=np.int64).reshape(-1, 1)

    combined_array = np.concatenate([timestamp_sequence, user_sequence_id, item_sequence_id, y_true, feature_sequence], axis=1)


    # 构建 DataFrame
    df_combined = pd.DataFrame(combined_array)
    df_combined.columns = ['timestamp', 'user', 'item', 'label'] + [f'feat_{i}' for i in
                                                                    range(feature_sequence.shape[1])]
    scaler = MinMaxScaler()
    df_combined['timestamp'] = scaler.fit_transform(df_combined[['timestamp']])
    # 添加 batch_id（按行数分批）
    total_rows = len(df_combined)
    batch_size = args.batch_size
    df_combined['batch_id'] = np.arange(total_rows) // batch_size
    print(df_combined)
    # 分组
    batch_groups = df_combined.groupby('batch_id')
    # 为每个时间段构建超图数据
    data_per_timeid = []
    for batch_id, group in batch_groups:
        # group 是该时间段下的子 dataframe
        data = process_batch_data(group)
        data_per_timeid.append(data)

    return data_per_timeid, len(batch_groups)



def process_batch_data( df):

    # print(df)
    df = df.sort_values(by="user")
    # print(df)
    # 给每一行分配一个唯一的边 ID
    df['edge_id'] = np.arange(len(df))

    # 为具有相同 item 的边分配相同的超边 ID
    item_to_hyperedge_id = {item: idx for idx, item in enumerate(df['user'].unique())}
    df['hyperedge_id'] = df['user'].map(item_to_hyperedge_id)

    # 构造 edge_index: shape = (2, num_edges)
    edge_index = np.stack([df['edge_id'].values, df['hyperedge_id'].values], axis=0)
    # print(edge_index)
    # print(len(df))

    # 对每个超边分组计算相似度矩阵
    grouped = df.groupby('user')
    edge_weights = []
    edge_attr = []
    # print(df)
    for item_id, group in grouped:
        # 取该组的特征矩阵 (num_edges_in_group, num_features)
        features = group.iloc[:, 4:-3].values
        # print(features.shape)
        if features.shape[0] == 1:
            # 只有一个节点，权重矩阵为1
            sim_matrix = torch.tensor([[1.0]])
            norm_sim_matrix = sim_matrix.clone()
        else:
            # 计算余弦相似度矩阵
            sim_matrix_np = cosine_similarity(features)
            sim_matrix = torch.tensor(sim_matrix_np, dtype=torch.float32)

            # 归一化：每行除以该行总和
            row_sum = sim_matrix.sum(dim=1, keepdim=True)
            norm_sim_matrix = sim_matrix / row_sum

        # hyperedge_id = group['hyperedge_id'].iloc[0]
        edge_weights.append(norm_sim_matrix)
        edge_attr.append(sim_matrix)

    # print(edge_attr.size())

    hyperedge_weights = [tensor.mean().item() for tensor in edge_weights]
    # print(len(edge_weights))
    # print(df['hyperedge_id'].values.max())

    features = df.iloc[:, 4:-3].values
    features_with_timestamp = np.hstack([features, df['timestamp'].values.reshape(-1, 1) ])
    features_tensor = torch.tensor(features_with_timestamp, dtype=torch.float32)
    # print(features_tensor.size())
    labels = df['label'].values
    labels_tensor = F.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=2).float()
    # print(labels_tensor.size())
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).contiguous()
    # print(edge_index_tensor.size())
    hyperedge_weights = torch.tensor(hyperedge_weights, dtype=torch.float32)
    hyperedge_weights_tensor = torch.nan_to_num(hyperedge_weights, nan=0.0)
    # print(hyperedge_weights_tensor.size())

    edge_attr_vectors = [x.flatten() for x in edge_attr]
    padded_edge_attr = rnn_utils.pad_sequence(edge_attr_vectors, batch_first=True, padding_value=0.0)
    # print(padded_edge_attr.size())

    data = Data(x=features_tensor,
                edge_attr=padded_edge_attr,
                edge_index=edge_index_tensor,
                y=labels_tensor,
                edge_weights=hyperedge_weights_tensor)

    return data







import sqlite3
import pandas as pd
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import scipy.sparse as sp

pd.set_option('display.max_columns', None)



def process_sniffer_data(DB_PATH, BATCH_SIZE):
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch all the relevant data
    cursor.execute('''SELECT data_packet_id, sourceID, snifferID, forwardCount, 
                      sourceCount, ackCount, routeReqCount, routeRepCount, lastRSSI 
                      FROM SnifferTable WHERE sourceID != 1 AND sourceID <= 11''')
    rows = cursor.fetchall()
    conn.close()

    # Convert to a DataFrame for easier processing
    columns = ["data_packet_id", "sourceID", "snifferID", "forwardCount",
               "sourceCount", "ackCount", "routeReqCount", "routeRepCount", "lastRSSI"]
    df = pd.DataFrame(rows, columns=columns)

    # Drop rows if not a multiple of the batch size
    df = df.iloc[:len(df) - (len(df) % BATCH_SIZE)]

    # Split into batches
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

    # Process each batch
    processed_batches = []
    for batch in batches:
        # Keep only the rows with the maximum data_packet_id for each (snifferID, sourceID) pair
        batch = batch.loc[batch.groupby(["snifferID", "sourceID"])['data_packet_id'].idxmax()]
        processed_batches.append(batch)

    return processed_batches


def process_realworld_batch(df):
    import torch
    import networkx as nx
    import numpy as np
    from torch_geometric.data import Data
    from sklearn.metrics.pairwise import cosine_similarity
    import torch.nn.functional as F

    # Sort by source node ID
    df = df.sort_values(by="sourceID")

    # Extract feature groups
    group_1 = torch.tensor(df[['forwardCount', 'sourceCount', 'ackCount', 'routeReqCount', 'routeRepCount', ]].values,
                           dtype=torch.float)
    group_2 = torch.tensor(df[['lastRSSI']].values, dtype=torch.float)
    node_features = torch.cat([group_1, group_2], dim=1)
    num_feature1 = group_1.size(1)

    # Create directed graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['sourceID'], row['snifferID'])

    # Get adjacency matrix
    adj = nx.adjacency_matrix(G)

    # Create transition matrix T
    T = create_transition_matrix(adj)
    T = sparse_mx_to_torch_sparse_tensor(T)

    # 步骤3：构建超边（每个目标节点（被监听节点）对应一个超边，包含所有与该节点相关的源节点）
    hyperedges = {}  # 用字典存储每个目标节点对应的所有源节点（超边）

    # 步骤4：为每条边分配唯一的ID
    edge_map = {}  # 用于存储边的唯一ID
    edge_id_counter = 0  # 边ID计数器
    # num_edges = len(df)  # 边的数量
    # edge_adjacency = np.zeros((num_edges, num_edges))  # 初始化邻接矩阵

    for _, row in df.iterrows():
        listener_node = row['snifferID']  # 监听节点
        src_node = row['sourceID']  # 被监听节点
        edge = (src_node, listener_node)  # 边由被监听节点和监听节点组成

        if edge not in edge_map:
            edge_map[edge] = edge_id_counter  # 为该边分配唯一ID
            edge_id_counter += 1

        # 将监听节点加入到对应的被监听节点的超边中
        if src_node not in hyperedges:
            hyperedges[src_node] = []
        hyperedges[src_node].append(edge_map[edge])  # 将边的ID加入超边中

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

    # 初始化标签，全为 0
    labels = torch.zeros(len(hyperedge_weights), dtype=torch.long)

    positive_indices = [9, 6]
    labels[positive_indices] = 1

    # 自动计算类别数量（0 和 1）
    num_classes = labels.max().item() + 1

    # 进行独热编码
    labels_tensor = F.one_hot(labels, num_classes=num_classes).float()



    # Create PyG Data object
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = edge_index[[1, 0]]
    hypernode_features = torch.stack(hypernode_features)
    edge_ids = torch.tensor(hyper_edge_ids, dtype=torch.long)
    hyperedge_weights = torch.tensor(hyperedge_weights)

    adj = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0])))

    return Data(x=hypernode_features[:, :-1], y=labels_tensor, edge_attr=hypernode_features[:, -1:],
                edge_index=edge_index, edge_ids=edge_ids, edge_weights=hyperedge_weights), T, adj


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


def process_realworld_data(DB_PATH, BATCH_SIZE):
    processed_data = process_sniffer_data(DB_PATH, BATCH_SIZE)
    # print(f"Processed {len(processed_data)} batches.")
    # print(processed_data[1])
    data_per_timeid = []
    adj_per_timeid = []
    T_per_timeid = []

    for i, batch in enumerate(processed_data):
        data, T, adj = process_realworld_batch(batch)
        data_per_timeid.append(data)
        adj_per_timeid.append(adj)
        T_per_timeid.append(T)
    return data_per_timeid, adj_per_timeid, T_per_timeid

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
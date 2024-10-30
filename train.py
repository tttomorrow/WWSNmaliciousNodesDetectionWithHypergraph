import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict
import numpy as np
import untils as lib
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, recall_score, precision_recall_curve, \
    f1_score, precision_score
from focalloss import FocalLoss
import model
import torch
from torch import nn, autograd
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from tqdm.auto import trange
from get_data import load_network


# %%
autograd.set_detect_anomaly(True)
# 初始化参数
parser = argparse.ArgumentParser()
parser.add_argument('--network', default=2, type=int, help='Name of the network/dataset')
parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--head', default=2, type=int)
parser.add_argument('--embedding_dim', default=8, type=int)
parser.add_argument('--layer', default=1, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--noise', default=0.3, type=float)
parser.add_argument('--alpa', default=0.2, type=float)
parser.add_argument('--anomaly', default=0.01, type=float)
args = parser.parse_args()
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

# if args.train_proportion > 0.8:
#     sys.exit('Training sequence proportion cannot be greater than 0.8.')

datapath = "../dataset/" + data_name + "/" + data_name + "_n_" + str(args.noise) + '0' + str(int(args.anomaly * 100)) + '.csv'
train_proportion = 0.7
torch.backends.cuda.max_split_size_mb = 1024
# %%

gpu = 0
# if gpu == -1:
#     gpu = lib.select_free_gpu()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# %%

# 加载数据集
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, y_true] = load_network(datapath)
# print(feature_sequence[1])
num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1  # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true) / (1.0 + sum(y_true))
# +1 in denominator in case there are no state change labels, which will throw an error.
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (
    num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# %%

# 生成batch时间长短
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500
batch_size = args.batch_size
# 初始化T-Batch相关的数据结构
tbatch_start_time = None
current_tbatch = []  # 存储当前T-Batch的交互数据
tbatch_list = []  # 存储所有T-Batch
interaction_ids = []
superedges = {}  # 存储超边信息的字典
current_size = 0
interaction_counter = 0
for timestamp, user_id, item_id in zip(
        timestamp_sequence, user_sequence_id, item_sequence_id
):
    interaction_id = interaction_counter
    interaction_counter += 1
    interaction_ids.append(interaction_id)

    # 如果当前交互的时间戳超过了T-Batch的时间跨度，则创建新的T-Batch
    if current_size > batch_size:
        # 添加当前T-Batch到T-Batch列表
        tbatch_list.append(current_tbatch)
        # 重置当前T-Batch和T-Batch的起始时间
        current_size = 0
        current_tbatch = []
        # tbatch_start_time = timestamp
    current_size = current_size + 1
    # 将当前交互添加到当前T-Batch中
    current_tbatch.append({
        'interaction_id': interaction_id,
        'user_id': user_id,
        'item_id': item_id,
        'timestamp': timestamp,
    })
# print(len(tbatch_list))
hypergraph_edge_data = []
hypergraph_inter_data = []
user_to_hyperedge = {}
hyperedge_ids = []
hyperinteraction_ids = []
hyperedge_counter = 0



# 设置训练集，测试集
train_end_idx = validation_start_idx = int(len(tbatch_list) * train_proportion)
test_start_idx = int(len(tbatch_list) * (train_proportion))
test_end_idx = int(len(tbatch_list) * (1))
# print(len(tbatch_list))
#
# print(test_start_idx)
# print(test_end_idx)
hypergraph_list = []
fea_list = []
timestamp_list = []
user_sequence_id_list = []
interaction_list = []
indices_list = []
# 遍历每个超图,构建数据集
data_path = './dataset/' + data_name + '/'
hypergraph_list_path = data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'hypergraph_list.pkl'
fea_list_path = data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'fea_list.pkl'

# 查询是否存在当前数据集的超图数据
if os.path.exists(hypergraph_list_path):
    if os.path.exists(fea_list_path):
        print("文件存在。")
        with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'hypergraph_list.pkl', 'rb') as f:
            hypergraph_list = pickle.load(f)
        with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'fea_list.pkl', 'rb') as f:
            fea_list = pickle.load(f)
        with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'timestamp_list.pkl', 'rb') as f:
            timestamp_list = pickle.load(f)
        with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'interaction_list.pkl', 'rb') as f:
            interaction_list = pickle.load(f)
        with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'indices_list.pkl', 'rb') as f:
            indices_list = pickle.load(f)
else:
    # 遍历每个T-Batch，构建超图
    for tbatch in tbatch_list:
        # 创建一个列表来存储当前T-Batch中的超边
        current_hyperedges = []
        current_interaction = []
        # 存储原始顺序
        # print(tbatch)
        # current_indices = list(range(len(tbatch)))
        # 遍历当前T-Batch中的交互数据
        for interaction in tbatch:
            # print(interaction)
            user_id = interaction['user_id']
            interaction_id = interaction['interaction_id']
            # 记录当前顺序

            # 检查用户是否已经存在于超边字典中
            if user_id in user_to_hyperedge:
                # 如果存在，将当前交互添加到对应的超边中
                hyperedge_id = user_to_hyperedge[user_id]
            else:
                # 如果不存在，创建一个新的超边ID
                hyperedge_id = hyperedge_counter
                hyperedge_counter += 1
                # 将新的超边ID映射到用户ID
                user_to_hyperedge[user_id] = hyperedge_id
                # current_hyperedges.append(hyperedge_id)
                # current_interaction.append(interaction['user_id'])
            # 记录超边ID
            # 添加当前交互的超边索引
            current_hyperedges.append(hyperedge_id)
            current_interaction.append(interaction['item_id'])
            # print(current_interaction)
            # print(current_hyperedges)
        # 将当前T-Batch的超边索引添加到超图数据中
        hypergraph_edge_data.append(current_hyperedges)
        hypergraph_inter_data.append(current_interaction)
        # indices_list.append(current_indices)



    # 将超边数据转换为 PyTorch 张量
    timestamp_sequence_array = np.array(timestamp_sequence)
    min_time = timestamp_sequence_array.min()
    max_time = timestamp_sequence_array.max()
    timestamp_sequence_array = (timestamp_sequence_array - min_time) / (max_time - min_time)


    # 保存超图数据
    with trange(len(hypergraph_inter_data)) as progress_bar:
        for i in progress_bar:
            progress_bar.set_description(
                'Processed %d of %d batches ' % (i, len(hypergraph_inter_data)))
            time.sleep(0.001)
            current_hypergraph = []
            current_fea = []
            current_timestamp = []
            # current_ind = indices_list[i]
            interaction = np.array(hypergraph_inter_data[i]).astype(int)
            hyperedge_ids = np.array(hypergraph_edge_data[i]).astype(int)
            # print(interaction)
            # print(hyperedge_ids)
            sorted_indices = np.argsort(hyperedge_ids)
            sorted_hyperedge_ids = hyperedge_ids[sorted_indices]
            sorted_interaction = interaction[sorted_indices]

            original_indices = np.argsort(sorted_indices)

            # print(sorted_indices)
            # print(original_indices)

            indices_list.append(original_indices)

            # print(sorted_hyperedge_ids)
            # print(sorted_interaction)
            current_hypergraph = np.vstack((sorted_interaction, sorted_hyperedge_ids))
            torch_current_hypergraph = torch.tensor(current_hypergraph, dtype=torch.int64)
            min_value = torch_current_hypergraph[0].min()
            torch_current_hypergraph[0] = torch_current_hypergraph[0] - min_value
            feature_array = np.array(feature_sequence)
            current_fea = torch.tensor(feature_array[sorted_interaction], dtype=torch.float32)
            current_timestamp = torch.tensor(timestamp_sequence_array[sorted_interaction], dtype=torch.float32).reshape(
                -1, 1)
            hypergraph_list.append(torch_current_hypergraph)
            fea_list.append(current_fea)
            timestamp_list.append(current_timestamp)
            interaction_list.append(sorted_interaction)
    # print(hypergraph_list)
    # print(interaction_list)
    with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'hypergraph_list.pkl', 'wb') as f:
        pickle.dump(hypergraph_list, f)
    with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'fea_list.pkl', 'wb') as f:
        pickle.dump(fea_list, f)
    with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'timestamp_list.pkl', 'wb') as f:
        pickle.dump(timestamp_list, f)
    with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'interaction_list.pkl', 'wb') as f:
        pickle.dump(interaction_list, f)
    with open(data_path + str(batch_size) + str(args.noise) + str(args.anomaly) + 'indices_list.pkl',
              'wb') as f:
        pickle.dump(indices_list, f)

# weight = torch.Tensor([1, true_labels_ratio]).cuda()
# crossEntropyLoss = nn.BCEWithLogitsLoss()
# loss_fun = model.BinaryCrossEntropyWithThreshold()
# MSELoss = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
focalLoss = FocalLoss(2)

# 设置训练参数
learning_rate = args.lr
embedding_dim = args.embedding_dim
l = args.layer
num_head = args.head
dropout = args.dropout
train_model = model.HypergraphTemporalModel(num_features, embedding_dim, l, num_head, dropout).cuda()
print(train_model)

# detect_optimizer = torch.optim.Adam(detect_model.parameters(), lr=0.01)

# optimizer = torch.optim.SGD([{'params': train_model.parameters(), 'lr': 0.001}],
#                             lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam([{'params': train_model.parameters(), 'lr': 0.001}],
                             lr=learning_rate)

# 设置训练轮数
epochs = 100
# 初始化数据
counter = 0
best_auc = 0.0
class_out = []
print("*** Training ***")
total_loss, loss = 0, 0
progress_bar1 = trange(epochs, ncols=100)

# with trange(epochs, ncols=100, disable=False) as progress_bar1:
for ep in progress_bar1:
    # progress_bar1.update(1)

    if ep == 0:
        last_auc = 0
        current_auc = 0
    else:
        last_auc = current_auc
        current_auc = auc(fpr, tpr)

    if current_auc > best_auc:
        best_auc = current_auc
    if current_auc < last_auc:
        counter = counter + 1
    else:
        counter = 0

    # if counter >= 20:
    #     print("Training terminated due to continuous AUC decrease. Best AUC: %f" % best_auc)
    #     break

    progress_bar2 = trange(train_end_idx, ncols=100, leave=False)
    progress_bar1.set_description(
        'Train auc: %f, loss : %f, training epoch %d of %d' % (current_auc, total_loss, ep + 1, epochs))
    total_loss, loss = 0, 0
    total_loss1, loss1 = 0, 0
    total_loss2, loss2 = 0, 0
    # with trange(train_end_idx, ncols=80, disable=False) as progress_bar2:
    for j in progress_bar2:
        progress_bar2.set_description('Processed %dth tbatch' % j)
        # time.sleep(0.1)
        current_hypergraph = hypergraph_list[j]
        current_interaction = interaction_list[j]
        # print("*****************")

        current_class_out = train_model.forward(fea_list[j], timestamp_list[j], current_hypergraph.cuda(), l, indices_list[j])

        # print(current_class_out)
        current_y_true = torch.Tensor(y_true)[current_interaction].cuda()
        # print(current_y_true)
        opposite_tensor = 1 - current_y_true
        combine_y = torch.stack((opposite_tensor, current_y_true))
        loss_input = torch.transpose(combine_y, 0, 1)

        loss1 = focalLoss(current_class_out[:,1], loss_input[:,1])
        # loss2 = criterion_class(loss_input, current_class_out)
        # loss = (1 * loss1) + (args.alpa * loss2.item())
        loss = loss1
        # print(loss)

        class_out.append(current_class_out)
        total_loss = total_loss + loss


        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(total_loss)


    # 测试模型
    train_model.eval()
    all_lables = []
    all_predicted = []
    for i in range(test_start_idx, test_end_idx):
        with torch.no_grad():
            current_interaction = interaction_list[i]
            current_hypergraph = hypergraph_list[i]

            class_output = train_model.forward(fea_list[i], timestamp_list[i], current_hypergraph.cuda(), l, indices_list[i])

            current_y_true = torch.Tensor(y_true)[current_interaction]
            predicted_probabilities_test = torch.softmax(class_output, dim=1).detach().cpu().numpy()
            true_labels_test = current_y_true.long().detach().cpu().numpy()  # true_labels_test
            # print("true_labels_test:", true_labels_test)
            current_true_labels_test = true_labels_test
            current_predicted_probabilities_test = predicted_probabilities_test[:, 1].flatten()
            # print("predicted_probabilities_test:", predicted_probabilities_test.flatten())

            all_lables = np.concatenate((all_lables, current_true_labels_test))
            all_predicted = np.concatenate((all_predicted, current_predicted_probabilities_test))

    fpr, tpr, thresholds = roc_curve(all_lables, all_predicted, pos_label=1)
    # thresholds[np.isinf(thresholds)] = 1
    # new_thresholds = np.linspace(thresholds.min(), thresholds.max(), len(all_predicted))

    precision, recall, _ = precision_recall_curve(all_lables, all_predicted)
    auc_pr = auc(recall, precision)
    # plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title(f'Precision-Recall curve: AUC-PR={auc_pr:.2f}')
    # plt.show()

    mapped_all_predicted = np.where(all_predicted >= 0.5, 0, 1)
    precision = precision_score(all_lables, mapped_all_predicted, zero_division=0)
    recall = recall_score(all_lables, mapped_all_predicted, zero_division=0)
    f1 = f1_score(all_lables, mapped_all_predicted, zero_division=0)
    report = classification_report(all_lables, mapped_all_predicted, zero_division=0)
    with open(data_path + "PR-diff_noise-" + str(args.noise) + ", anomaly-" + str(args.anomaly) + ", batch_size-" + str(batch_size) +
              ", learning_rate-" + str(learning_rate) + ", embedding_dim-" + str(embedding_dim) + ", l-" +
              str(l) + ", num_head-" + str(num_head) + ", dropout-" + str(dropout) + ", alpa-" + str(args.alpa) + ".txt", 'a') as file:

        file.write(f'#############################################################################'
                   f'\nEpoch {ep + 1}.'
                   f'\nTEST auc: {auc(fpr, tpr)}, \nPrecision: {precision}, \nRecall: {recall}, '
                   f'\nAUC_PR: {auc_pr},'
                   f'\nF1: {f1}.\n'
                   f'{report}\n')
        file.flush()
        # file.write(f'Epoch {ep + 1}, VAL auc: {auc(fpr1, tpr1)}\n')
        # file.flush()

    train_model.train()
    file.close()

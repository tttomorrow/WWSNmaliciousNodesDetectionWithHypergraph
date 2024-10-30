from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale


# LOAD THE NETWORK
def load_network(datapath, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.
    '''
    # if args.network == 1:
    #     network = 'wikipedia'
    # if args.network == 2:
    #     network = 'mooc'
    # if args.network == 3:
    #     network = 'reddit'
    # if args.network == 4:
    #     network = 'lastfm'
    # print(args)
    # if args.network == 1:
    #     args.datapath = "../dataset/wikipedia/wikipedia.csv"
    # if args.network == 2:
    #     args.datapath = "../dataset/mooc/mooc.csv"
    # if args.network == 3:
    #     args.datapath = "../dataset/reddit/reddit.csv"
    # if args.network == 4:
    #     args.datapath = "../dataset/reddit/reddit.csv"
    # if args.network == 5:
    #     args.datapath = "../data/eucore/eucore.csv"
    # # network = args.network
    # # print(args.datapath)
    # # datapath = args.datapath
    network = 2

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    # 打开数据文件，逐行读取
    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath, "r")
    f.readline()  # 读取文件的第一行，通常包含标题或注释，跳过它
    for cnt, l in enumerate(f):
        # 格式：user, item, timestamp, state label, feature list
        ls = l.strip().split(",")  # 将每一行按逗号分割成多个字段
        user_sequence.append(ls[0])  # 提取用户序列
        item_sequence.append(ls[1])  # 提取物品序列
        if start_timestamp is None:
            start_timestamp = float(ls[2])  # 提取起始时间戳（如果尚未提取）
        timestamp_sequence.append(float(ls[2]) - start_timestamp)  # 计算时间戳相对于起始时间的差值
        y_true_labels.append(int(ls[3]))  # 标签=1表示状态变化，否则为0
        feature_sequence.append(list(map(float, ls[4:])))  # 提取特征列表并转换为浮点数
    f.close()  # 关闭文件

    user_sequence = np.array(user_sequence)  # 将用户序列转换为NumPy数组
    item_sequence = np.array(item_sequence)  # 将物品序列转换为NumPy数组
    timestamp_sequence = np.array(timestamp_sequence)  # 将时间戳序列转换为NumPy数组

    # 格式化物品序列
    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]  # 将物品序列转换为对应的ID序列

    # 格式化用户序列
    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]  # 将用户序列转换为对应的ID序列

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")

    # 返回所提取的数据作为结果
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, item2id,
            item_sequence_id, item_timedifference_sequence,
            timestamp_sequence, feature_sequence, y_true_labels]
    # user2id：一个字典，将用户ID映射到用户节点ID。用户节点ID是整数，用于在网络中唯一标识每个用户。
    #
    # user_sequence_id：一个列表，包含了用户序列中每个用户的用户节点ID。它是根据user2id字典生成的。
    #
    # user_timedifference_sequence：一个列表，包含了用户序列中每个用户之间的时间差。它表示用户在时间上的行为间隔。
    #
    # user_previous_itemid_sequence：一个列表，包含了用户序列中每个用户之前与之相关的物品的节点ID。它表示每个用户在之前与哪些物品有关联。
    #
    # item2id：一个字典，将物品ID映射到物品节点ID。物品节点ID是整数，用于在网络中唯一标识每个物品。
    #
    # item_sequence_id：一个列表，包含了物品序列中每个物品的物品节点ID。它是根据item2id字典生成的。
    #
    # item_timedifference_sequence：一个列表，包含了物品序列中每个物品之间的时间差。它表示物品在时间上的出现间隔。
    #
    # timestamp_sequence：一个列表，包含了每个事件的时间戳，通常是相对于某个起始时间的差值。
    #
    # feature_sequence：一个列表，包含了每个事件的特征向量。这些特征向量可能包含了关于事件的其他信息。
    #
    # y_true_labels：一个列表，包含了每个事件的真实标签。在这里，标签为1表示状态变化，0表示没有状态变化。
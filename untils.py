from collections import defaultdict

import gpustat
import numpy as np
import torch


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




# 创建一个函数来构建超图
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





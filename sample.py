import pandas as pd
import numpy as np

import time
import random

# from tqdm import tqdm
# from tqdm._tqdm import trange
#
# pbar = tqdm(["a", "b", "c", "d"])
# for char in pbar:
#     pbar.set_description("Processing %s" % char)
# # 1. 读取原始CSV文件
#
# 读取CSV文件\
# a：噪声比率；b：负样本比率
# b = 0.1
for i in range(2):
    # if i == 0:
    #     file = "digg"
    # elif i == 1:
    #     file = "uci"
    if i == 0:
        file = "mooc"
    elif i == 1:
        file = "reddit"
    elif i == 2:
        file = "eucore"
    elif i == 3:
        file = "wikipedia"
    for k in range(3):
        if k == 0:
            b = 0.01
        elif k == 1:
            b = 0.05
        elif k == 2:
            b = 0.1
        for j in range(5):
            if j == 0:
                a = 0.0
            elif j == 1:
                a = 0.1
            elif j == 2:
                a = 0.2
            elif j == 3:
                a = 0.3
            elif j == 4:
                a = 0.4

            # print('i:%f' % (i))
            # print('j:%f' % (j))
            df = pd.read_csv('../dataset/' + file + '/' + file + '.csv', skiprows=1, header=None)
            # print(df.shape)

            n = df.shape[0]
            if file == "digg" or file == "uci" or file == "eucore":
                for i in range(4):
                    new_column = np.zeros(len(df))
                    column_name = i + 4
                    df[column_name] = new_column
            # print(df)`
            if file == "digg" or file == "uci":
                df.loc[:, 3] = 1 - df.loc[:, 3]
                # print("1")

            # print(file)
            # print(df)

            num_features = df.shape[1] - 4

            #
            state_label_1 = df[df[3] == 1]
            state_label_0 = df[df[3] == 0]

            edges = df.loc[[0, 1]].values
            edges_set = set(map(tuple, edges))
            # print(edges_set)
            # print(state_label_1)

            # 设置负样本比例
            num_negative_samples = int(len(state_label_0) * b)
            print(num_negative_samples)
            # 负采样
            negative_samples = []
            while len(negative_samples) < num_negative_samples:
                # Randomly select a user and item
                user = random.choice(df[0])
                item = random.choice(df[1])
                timestamp = random.choice(df[2])
                # Check if the edge (user, item) does not exist in positive edges
                if (user, item) not in edges_set:
                    # Generate random features for the negative sample
                    random_features = [random.random() for _ in range(num_features)]
                    sample = [user, item, timestamp, 1]
                    sample.extend(random_features)
                    negative_samples.append(sample)

            # list转数组
            negative_samples = pd.DataFrame(negative_samples)
            # negative_samples = np.array(negative_samples)
            # 负采样加入数据集
            # print(negative_samples.shape[0])
            combined_dataset = pd.concat([df, negative_samples])
            # print(combined_dataset.shape[0])
            # df = df.values
            # combined_dataset = np.concatenate((df, negative_samples), axis=0)
            # sorted_indices = np.argsort(combined_dataset[:, 2])
            # combined_dataset = combined_dataset[sorted_indices]
            # combined_dataset = combined_dataset.sort_values(by=2)

            if a > 0:
                # 注入噪声数据

                if file == "digg" or file == "uci" or file == "eucore":
                    num_edges_to_select = int(n * a)

                    # 删除构建缺失边
                    rows_to_delete = np.random.choice(combined_dataset.index, int(num_edges_to_select / 2),
                                                      replace=False)
                    combined_dataset = combined_dataset.drop(rows_to_delete)

                    # 构建缺陷边，特征为随机值
                    # 随机选择边的索引
                    selected_indices = np.random.choice(combined_dataset.index, int(num_edges_to_select / 2),
                                                        replace=False)
                    # 复制边并加入噪声
                    noise_row = combined_dataset.loc[selected_indices].copy()
                    noise_row.loc[:, 4:] = np.random.rand(*noise_row.loc[:, 4:].shape)
                    # 删去噪声边对应的原始边
                    combined_dataset = combined_dataset.drop(selected_indices)

                    # 噪声集加入原始数据
                    combined_dataset = pd.concat([combined_dataset, noise_row])
                    combined_dataset = combined_dataset.sort_values(by=2)
                else:
                    num_edges_to_select = int(n * a)

                    # 删除构建缺失边
                    rows_to_delete = np.random.choice(combined_dataset.index, int(num_edges_to_select / 2),
                                                      replace=False)
                    combined_dataset = combined_dataset.drop(rows_to_delete)

                    # 构建缺陷边，特征为0
                    # 随机选择边的索引
                    selected_indices = np.random.choice(combined_dataset.index, int(num_edges_to_select / 2),
                                                        replace=False)
                    # 复制边并加入噪声
                    noise_row = combined_dataset.loc[selected_indices].copy()
                    noise_row.loc[:, 3:] = 0
                    # 删去噪声边对应的原始边
                    combined_dataset = combined_dataset.drop(selected_indices)

                    # 噪声集加入原始数据
                    combined_dataset = pd.concat([combined_dataset, noise_row])
                    combined_dataset = combined_dataset.sort_values(by=2)


            # print(combined_dataset)
            # 或者保存为新的CSV文件combined_data
            combined_dataset.to_csv('../dataset/' + file + '/' + file + '_n_' + str(a) + '0' + str(int(b * 100)) + '.csv',
                                    index=False, header=False)
            # print(combined_data.shape)

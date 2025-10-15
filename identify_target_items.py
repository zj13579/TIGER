'''
将异常样本加入到真实数据集中进行模型训练，观察前后推荐结果的变化，判断目标商品
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from RSs.LFM import LFM
from collections import defaultdict
from utils.functions import *
from collections import defaultdict
import math

np.random.seed(seed=0)  # 设置的随机数种子
torch.manual_seed(seed=0)  # 设置torch上面的随机数种子np上面


class IdentifyTargets:
    def __init__(self, dataset_name, num_items, lat_fac, lr_it, num_epochs_it, lambda_value_it):
        self.dataset_name = dataset_name
        self.num_items = num_items

        self.lat_fac = lat_fac
        self.lr_it = lr_it
        self.num_epochs_it = num_epochs_it
        self.lambda_value_it = lambda_value_it

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.df = pd.read_csv(f'./datasets/{self.dataset_name}/real_dataset.txt', sep=' ', header=None, names=['user', 'item', 'rating'])
        self.df_abnormal = pd.read_csv(f'./datasets/{self.dataset_name}/abnormal_samples.txt', sep=' ', header=None, names=['user', 'item', 'rating'])  # 方案一得到的异常样本

    def attack_before(self):

        print('----------攻击前的模型训练----------')
        users = self.df['user'].unique()
        num_users = len(users)
        path = f'./datasets/{self.dataset_name}/real_dataset.txt'
        model_LFM = LFM(num_users, self.num_items, self.lat_fac, device=self.device)
        model_LFM.train_model(path, self.lr_it, self.num_epochs_it, self.lambda_value_it)
        _, _, ratings_pred, _ = model_LFM(self.lambda_value_it)

        user_train_ids, item_train_ids, train_data = read_data(num_users, self.num_items, path=path)
        ratings_indices_dict, filtered_indices = obtain_filtered_indices(user_train_ids, item_train_ids, defaultdict,
                                                                         self.num_items)
        origin_Top_K_list, origin_rec_list = obtain_rec_list(ratings_pred, filtered_indices, defaultdict, num_users,
                                                             ratings_indices_dict, K=10)

        return origin_Top_K_list, origin_rec_list

    def attack_after(self):

        print('----------攻击后的模型训练----------')
        # 把异常用户数据集合并到真实数据集中
        max_user = self.df['user'].max()
        unique_users = self.df_abnormal['user'].unique()
        user_mapping = {old_user: new_user for new_user, old_user in enumerate(unique_users, start=max_user + 1)}
        self.df_abnormal['user'] = self.df_abnormal['user'].map(user_mapping)
        df_merged = pd.concat([self.df, self.df_abnormal], ignore_index=True)

        users_updated = df_merged['user'].unique()
        num_users_updated = len(users_updated)
        # 统一数据类型
        df_merged['user'] = df_merged['user'].astype(int)
        df_merged['item'] = df_merged['item'].astype(int)
        df_merged['rating'] = df_merged['rating'].astype(float)
        # 转成一个评分矩阵格式的张量
        rating_matrix = torch.zeros((num_users_updated, self.num_items), dtype=torch.float64).to(self.device)
        for row in df_merged.itertuples(index=False):
            rating_matrix[row.user, row.item] = row.rating

        model_LFM = LFM(num_users_updated, self.num_items, self.lat_fac, device=self.device)
        model_LFM.retrain_model(rating_matrix, self.lr_it, self.num_epochs_it, self.lambda_value_it)
        _, _, ratings_pred, _ = model_LFM(self.lambda_value_it)
        user_ids = torch.tensor(df_merged['user'].values, dtype=torch.long)
        item_ids = torch.tensor(df_merged['item'].values, dtype=torch.long)
        ratings_indices_dict, filtered_indices = obtain_filtered_indices(user_ids, item_ids, defaultdict,
                                                                         self.num_items)
        Top_K_list, rec_list = obtain_rec_list(ratings_pred, filtered_indices, defaultdict, num_users_updated,
                                               ratings_indices_dict, K=10)

        return Top_K_list, rec_list

    def calculate_rank_changes_with_freq(self, origin_rec_list, rec_list):

        all_item_changes = defaultdict(list)  # {商品id: [排名变化1, 排名变化2, ...]}
        item_user_count = defaultdict(int)  # {商品id: 出现的用户数}

        # 1. 收集每个商品的排名变化
        for origin_list, new_list in zip(origin_rec_list, rec_list):
            origin_pos = {item: idx for idx, item in enumerate(origin_list)}
            new_pos = {item: idx for idx, item in enumerate(new_list)}

            for item in origin_list:
                rank_change = origin_pos[item] - new_pos[item]
                all_item_changes[item].append(rank_change)
                item_user_count[item] += 1

        # 2. 计算平均排名变化
        avg_rank_change = {}
        for item, changes in all_item_changes.items():
            avg_rank_change[item] = round(sum(changes) / len(changes), 4)

        # 3. 数据向量化处理
        items = list(avg_rank_change.keys())
        rank_changes = np.array([avg_rank_change[item] for item in items])
        freqs = np.array([item_user_count[item] for item in items])

        # 4. 生成结果
        result = []
        for i, item in enumerate(items):
            result.append({
                'item_id': item,
                'user_count': item_user_count[item],
                'avg_rank_change': round(avg_rank_change[item], 4),
            })

        # 按score降序排序
        result.sort(key=lambda x: x['avg_rank_change'], reverse=True)

        return result

    def inference_target_items(self, target_item_list):
        _, origin_rec_list = self.attack_before()
        _, rec_list = self.attack_after()
        remained_rec_list = rec_list[:len(origin_rec_list)]  # 只保留真实用户商品推荐的排名变化
        suspicious_items = self.calculate_rank_changes_with_freq(origin_rec_list, remained_rec_list)
        top_target_ids = [v['item_id'] for v in suspicious_items[:100]]

        return top_target_ids
import torch
import torch.nn as nn
import torch.optim as optim
from utils.functions import read_data
import pandas as pd
import numpy as np
from utils.functions import *
torch.manual_seed(seed=0)

class MatrixFactorization:
    def __init__(self, n_items, latent_factors, dataset_name, device):
        self.n_items = n_items
        self.latent_factors = latent_factors
        self.dataset_name = dataset_name
        self.device = device

    def obtain_item_vectors(self, lr_v=0.05, epochs_v=5000):
        real_dat_path = f'./datasets/{self.dataset_name}/real_dataset.txt'
        data = pd.read_csv(real_dat_path, sep=' ', header=None, names=['user', 'item', 'rating'])
        users = data['user'].unique()
        mf = MF(len(users), self.n_items, self.latent_factors, self.device, item_vectors=None)
        mf.train_model(real_dat_path, lr_v, epochs_v)
        item_vectors = mf.item_embeddings.weight.data.cpu()  # 从训练好的模型中提取商品的向量化表示
        save_path = f'./results/{self.dataset_name}/item_vectors.pt'
        torch.save(item_vectors, save_path)  # 保存张量

        print('---------------商品向量已保存---------------\n')
        return item_vectors

    def construct_combined_dataset(self, attack_type):
        remained_dat_path = f'./datasets/{self.dataset_name}/remained_dataset.txt'
        data = pd.read_csv(remained_dat_path, sep=' ', header=None, names=['user', 'item', 'rating'])
        fake_profiles = np.load(f'./results/{self.dataset_name}/{attack_type}_fake_profiles.npy')  # shape: (n_fake_users, n_items)
        max_user_id = data['user'].max()
        n_fake_users, n_items = fake_profiles.shape

        # 转换 fake_profiles 为 DataFrame 格式（user, item, rating）
        fake_rows = []
        for i in range(n_fake_users):
            user_id = max_user_id + 1 + i
            for item_id in range(n_items):
                rating = fake_profiles[i, item_id]
                if rating > 0:  # 只保留非零评分
                    fake_rows.append([user_id, item_id, int(rating)])

        fake_df = pd.DataFrame(fake_rows, columns=['user', 'item', 'rating'])
        print(f'合成数据集中真实用户的数量为：{max_user_id + 1}个,虚假用户的数量为：{len(fake_profiles)}个')
        combined_dataset = pd.concat([data, fake_df], ignore_index=True)
        combined_dataset = combined_dataset.sort_values(by='user').reset_index(drop=True)
        combined_dataset.to_csv(f'./datasets/{self.dataset_name}/combined_dataset.txt', sep=' ', header=False, index=False)

        # 对合成数据集划分训练集和测试集
        combined_test_dataset = combined_dataset.groupby('user').sample(n=1, random_state=0)
        combined_test_dataset.to_csv(f'./datasets/{self.dataset_name}/combined_test_dataset.txt', sep=' ', header=False, index=False)
        combined_train_dataset = combined_dataset.drop(index=combined_test_dataset.index)
        combined_train_dataset = combined_train_dataset.sort_values(by='user').reset_index(drop=True)
        combined_train_dataset.to_csv(f'./datasets/{self.dataset_name}/combined_train_dataset.txt', sep=' ', header=False, index=False)

        print('---------------构造合成数据集已完成---------------\n')
        return combined_dataset, combined_train_dataset, combined_test_dataset

    def obtain_user_vectors(self, lr_u=0.1, epochs_u=2000):
        combined_tra_dat_path = f'./datasets/{self.dataset_name}/combined_train_dataset.txt'
        combined_train_dataset = pd.read_csv(combined_tra_dat_path, sep=' ', header=None, names=['user', 'item', 'rating'])
        users = combined_train_dataset['user'].unique()
        item_vectors = torch.load(f'./results/{self.dataset_name}/item_vectors.pt')  # 形状为 (num_items, latent_factors)
        model = MF(len(users), self.n_items, self.latent_factors, self.device, item_vectors=item_vectors)
        model.train_model(combined_tra_dat_path, lr_u, epochs_u)

        # 保存用户向量
        user_latent_features = model.user_embeddings.weight.data.cpu()
        torch.save(user_latent_features, f'./results/{self.dataset_name}/user_latent_features.pt')

        print('---------------用户隐特征向量已保存---------------')
        return user_latent_features
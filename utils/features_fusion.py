import pandas as pd
import numpy as np
import torch
import pickle
import os
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
from utils.functions import train_autoencoder
from utils.functions import FeatureAttentionFusion


class FeaturesFusion:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.pkl_path = f'./results/{self.dataset_name}/user_features.pkl'

    def _load_user_features(self):
        """读取 user_features.pkl 并解包"""
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"{self.pkl_path} 不存在，请先生成 user_features.pkl")
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)

        return (
            data['r_count'], data['r_mean'], data['r_var'],
            data['h_r_ratio'], data['l_r_ratio'],
            data['dev'], data['imp'], data['div'],
            data['sta_best'], data['sta_avg']
        )

    def method_1(self, pca_dimension, rating_features_only: bool = False):
        """方案一：拼接 + PCA"""

        (
            user_rating_count, user_rating_mean, user_rating_var,
            user_high_rating_ratio, user_low_rating_ratio,
            user_deviation, user_importance, user_diversity,
            user_stability_best, user_stability_avg
        ) = self._load_user_features()

        # === 1. 整理第一类特征（局部特征）: 构建 DataFrame
        df_local = pd.DataFrame({
            'user': list(user_rating_count.keys()),
            'rating_count': list(user_rating_count.values()),
            'rating_mean': list(user_rating_mean.values()),
            'rating_var': list(user_rating_var.values()),
            'high_rating_ratio': list(user_high_rating_ratio.values()),
            'low_rating_ratio': list(user_low_rating_ratio.values()),
        })
        df_local.set_index('user', inplace=True)

        # === 2. 整理第二类特征（全局特征）：评分偏离度 + 评分重要性（拼接 & PCA降维）
        user_global_features = {}
        for uid in user_deviation:
            dev_vec = user_deviation[uid]
            imp_vec = user_importance[uid]
            global_vec = dev_vec + imp_vec  # 长度为用户评分数量的两倍
            user_global_features[uid] = global_vec
        # 对齐成等长矩阵（用0填充，使所有用户特征长度一致）
        max_len = max(len(v) for v in user_global_features.values())
        X_global = []
        uid_list = []
        for uid, vec in user_global_features.items():
            padded = vec + [0.0] * (max_len - len(vec))  # 结合评分偏离度和评分重要性的特征定义，这里用0来填充也是比较合理的
            X_global.append(padded)
            uid_list.append(uid)
        X_global = np.array(X_global)
        # PCA降维到16维
        pca = PCA(n_components=pca_dimension, random_state=0)
        X_global_reduced = pca.fit_transform(X_global)
        df_global = pd.DataFrame(X_global_reduced, index=uid_list)

        # === 3. 整理第三类特征（行为偏好特征）：多样性 + 稳定性
        df_pref = pd.DataFrame({
            'user': list(user_diversity.keys()),
            'diversity': list(user_diversity.values()),
            'stability_best': list(user_stability_best.values()),
            'stability_avg': list(user_stability_avg.values()),
        })
        df_pref.set_index('user', inplace=True)

        # === 4. 合并所有特征
        print('-----采用(方案一)进行特征融合-----')
        # print(f"user_local_features：\n {df_local[:3]}")
        # print(f"user_global_features：\n {df_global[:3]}")
        # print(f"user_preference_features：\n {df_pref[:3]}")

        if rating_features_only:
            df_all = pd.concat([df_local, df_global], axis=1).sort_index()
        else:
            df_all = pd.concat([df_local, df_global, df_pref], axis=1).sort_index()
            # df_all = df_pref.sort_index()

        df_all.fillna(0.0, inplace=True)  # 有些用户可能少部分特征缺失，填充0
        user_behavior_features = torch.tensor(df_all.values, dtype=torch.float32)  # 转换成tensor的数据格式
        print("最终用户特征向量 shape:", user_behavior_features.shape)  # 最终用户特征向量 shape: [num_users, 24]
        torch.save(user_behavior_features, f'./results/{self.dataset_name}/user_behavior_features_1.pt')

        return user_behavior_features


    def method_2(self, rating_features_only: bool = False):
        """方案二：拼接 + AutoEncoder + Attention融合"""

        (
            user_rating_count, user_rating_mean, user_rating_var,
            user_high_rating_ratio, user_low_rating_ratio,
            user_deviation, user_importance, user_diversity,
            user_stability_best, user_stability_avg
        ) = self._load_user_features()

        # === 1. 第一类特征：
        df_local = pd.DataFrame({
            'rating_count': list(user_rating_count.values()),
            'rating_mean': list(user_rating_mean.values()),
            'rating_var': list(user_rating_var.values()),
            'high_rating_ratio': list(user_high_rating_ratio.values()),
            'low_rating_ratio': list(user_low_rating_ratio.values()),
        })
        user_local_features = {i: row.tolist() for i, row in df_local.iterrows()}

        # === 2. 第二类特征：
        user_global_features = {}
        for uid in user_deviation:
            dev_vec = user_deviation[uid]
            imp_vec = user_importance[uid]
            global_vec = dev_vec + imp_vec  # 长度为用户评分数量的两倍
            user_global_features[uid] = global_vec
        # 对第二类特征进行 padding
        global_features_tensor_list = [torch.tensor(v, dtype=torch.float32) for v in user_global_features.values()]
        padded_global_features = pad_sequence(global_features_tensor_list, batch_first=True)
        # 用 AutoEncoder 降维第二类特征
        user_global_features_encoded = train_autoencoder(
            {uid: vec.tolist() for uid, vec in zip(user_global_features.keys(), padded_global_features)})
        array_data = np.array([user_global_features_encoded[uid] for uid in user_global_features.keys()],
                              dtype=np.float32)

        # === 3. 第三类特征：
        df_pref = pd.DataFrame({
            'diversity': list(user_diversity.values()),
            'stability_best': list(user_stability_best.values()),
            'stability_avg': list(user_stability_avg.values()),
        })
        user_preference_features = {i: row.tolist() for i, row in df_pref.iterrows()}

        # === 4. 构造输入张量
        uids = list(user_global_features_encoded.keys())
        x1 = torch.tensor([user_local_features[uid] for uid in uids], dtype=torch.float32)
        x2 = torch.from_numpy(array_data)
        x3 = torch.tensor([user_preference_features[uid] for uid in uids], dtype=torch.float32)

        print('-----采用(方案二)进行特征融合-----')
        print(f"user_local_features：\n {x1[:3]}")
        print(f"user_global_features：\n {x2[:3]}")
        print(f"user_preference_features：\n {x3[:3]}")

        # === 5. 融合特征

        if rating_features_only:
            input_dim = x1.size(1) + x2.size(1)  # 比如 21
            fusion_model = FeatureAttentionFusion(input_dim)
            user_behavior_features = fusion_model(x1, x2)
        else:
            # input_dim = x1.size(1) + x2.size(1) + x3.size(1)  # 比如 24
            # fusion_model = FeatureAttentionFusion(input_dim)
            # user_behavior_features = fusion_model(x1, x2, x3)

            input_dim = x3.size(1)
            fusion_model = FeatureAttentionFusion(input_dim)
            user_behavior_features = fusion_model(x3)

        print("最终用户特征向量 shape:", user_behavior_features.shape)  # 最终用户特征向量 shape: [num_users, 64]
        torch.save(user_behavior_features, f'./results/{self.dataset_name}/user_behavior_features_2.pt')

        return user_behavior_features


    def method_3(self, rating_features_only: bool = False):
        """方案三：拼接 + AutoEncoder"""
        (
            user_rating_count, user_rating_mean, user_rating_var,
            user_high_rating_ratio, user_low_rating_ratio,
            user_deviation, user_importance, user_diversity,
            user_stability_best, user_stability_avg
        ) = self._load_user_features()

        # === 1. 整理第一类特征（局部特征）
        df_local = pd.DataFrame({
            'user': list(user_rating_count.keys()),
            'rating_count': list(user_rating_count.values()),
            'rating_mean': list(user_rating_mean.values()),
            'rating_var': list(user_rating_var.values()),
            'high_rating_ratio': list(user_high_rating_ratio.values()),
            'low_rating_ratio': list(user_low_rating_ratio.values()),
        })
        df_local.set_index('user', inplace=True)

        # === 2. 整理第二类特征（全局特征）：评分偏离度 + 评分重要性 → AutoEncoder降维
        user_global_features = {}
        for uid in user_deviation:
            dev_vec = user_deviation[uid]
            imp_vec = user_importance[uid]
            global_vec = dev_vec + imp_vec  # 长度为用户评分数量的两倍
            user_global_features[uid] = global_vec

        # 对第二类特征进行 padding（对齐到相同长度）
        global_features_tensor_list = [torch.tensor(v, dtype=torch.float32) for v in user_global_features.values()]
        padded_global_features = pad_sequence(global_features_tensor_list, batch_first=True)

        # 用 AutoEncoder 降维第二类特征
        user_global_features_encoded = train_autoencoder(
            {uid: vec.tolist() for uid, vec in zip(user_global_features.keys(), padded_global_features)}
        )
        array_data = np.array([user_global_features_encoded[uid] for uid in user_global_features.keys()],
                              dtype=np.float32)
        df_global = pd.DataFrame(array_data, index=user_global_features.keys())

        # === 3. 整理第三类特征（行为偏好特征）
        df_pref = pd.DataFrame({
            'user': list(user_diversity.keys()),
            'diversity': list(user_diversity.values()),
            'stability_best': list(user_stability_best.values()),
            'stability_avg': list(user_stability_avg.values()),
        })
        df_pref.set_index('user', inplace=True)

        # === 4. 合并所有特征
        print('-----采用(方案三)进行特征融合-----')

        if rating_features_only:
            df_all = pd.concat([df_local, df_global], axis=1).sort_index()
        else:
            df_all = pd.concat([df_local, df_global, df_pref], axis=1).sort_index()
            # df_all = df_pref.sort_index()

        df_all.fillna(0.0, inplace=True)
        user_behavior_features = torch.tensor(df_all.values, dtype=torch.float32)
        print("最终用户特征向量 shape:", user_behavior_features.shape)
        torch.save(user_behavior_features, f'./results/{self.dataset_name}/user_behavior_features_3.pt')

        return user_behavior_features

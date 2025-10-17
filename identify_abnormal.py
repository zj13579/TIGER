import torch
from sklearn.preprocessing import StandardScaler
import hdbscan
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from identify_target_items import IdentifyTargets
from collections import Counter
import random
from collections import Counter, defaultdict

random.seed(0)
np.random.seed(0)


class IdentifyAbnormal:
    def __init__(self, user_representations, dataset_name, num_items, lat_fac, lr_it, num_epochs_it, lambda_value_it):
        self.user_representations = user_representations
        self.dataset_name = dataset_name
        self.num_items = num_items

        self.lat_fac = lat_fac
        self.lr_it = lr_it
        self.num_epochs_it = num_epochs_it
        self.lambda_value_it = lambda_value_it

        self.data = pd.read_csv(f'./datasets/{self.dataset_name}/combined_dataset.txt', sep=' ', header=None,
                                names=['user', 'item', 'rating'])

        self.remained_data = pd.read_csv(f'./datasets/{self.dataset_name}/remained_dataset.txt', sep=' ', header=None,
                                         names=['user', 'item', 'rating'])
        self.num_real_users = self.remained_data['user'].max() + 1
        self.num_all_users = len(self.user_representations)
        self.fake_user_ids = list(range(self.num_real_users, self.num_all_users))

        self.thresholds = [0.85, 0.80, 0.75]

    def cluster(self):
        # === 1. 读入数据
        print(f'user_representations.shape:{self.user_representations.shape}')
        Z = self.user_representations.detach().cpu().numpy()  # 将原来的张量变成数组的数据类型
        scaler = StandardScaler()
        Z_std = scaler.fit_transform(Z).astype(np.float64)  # 标准化+固定精度

        # === 2. HDBSCAN聚类
        min_cluster_size = math.ceil(0.01 * len(Z_std))  # 这里设置总样本数的0.5%-2%作初始值
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=None,
                                    core_dist_n_jobs=1)  # min_samples：控制局部密度稳健性（建议与min_cluster_size同量级或稍小）
        labels = clusterer.fit_predict(Z_std)  # labels == -1 表示 noise
        unique_classes = np.unique(labels)
        print("找到的类别标签有:", unique_classes)
        print(f"总共有 {len(unique_classes)} 个不同的类别")

        return Z_std, labels

    def cal_abnormal_ratings(self, w):
        # === 计算异常评分
        '''
        通过构造score_noise、score_cluster_size、score_distance等指标，然后加权的方式计算得到总体异常评分
        '''
        # (1) 计算score_distance
        Z_std, labels = self.cluster()
        unique_labels = set(labels) - {-1}
        centers = {lab: Z_std[labels == lab].mean(axis=0) for lab in unique_labels}
        distances = np.zeros(len(Z_std))
        for i in range(len(Z_std)):
            lab = labels[i]
            if lab == -1:
                distances[i] = np.linalg.norm(Z_std[i] - Z_std.mean(axis=0))  # fallback
            else:
                distances[i] = np.linalg.norm(Z_std[i] - centers[lab])
        # 标准化距离到 z-score
        dist_z = (distances - distances.mean()) / (distances.std() + 1e-8)
        score_distance = (dist_z - dist_z.min()) / (dist_z.max() - dist_z.min())

        # (2) 计算score_cluster_size
        cluster_sizes = {lab: np.sum(labels == lab) for lab in unique_labels}
        score_cluster_size = np.zeros(len(Z_std))
        for i in range(len(Z_std)):
            lab = labels[i]
            if lab == -1:
                score_cluster_size[i] = 1.0
            else:
                s = cluster_sizes[lab]
                score_cluster_size[i] = 1 - np.log(s + 1) / np.log(len(Z_std) + 1)

        # (3) 计算score_noise
        score_noise = (labels == -1).astype(float)

        # (4) 计算综合得分（示例权重）
        score = w[0] * score_noise + w[1] * score_cluster_size + w[2] * score_distance

        # === 得到每个样本属于异常样本的概率，按照概率从高到低排序
        rounded_score = np.round(score, decimals=6)
        sorted_indices = np.argsort(rounded_score)[::-1]
        sorted_values = rounded_score[sorted_indices]
        score_dict = {i: (original_idx, value) for i, (original_idx, value) in
                      enumerate(zip(sorted_indices,
                                    sorted_values))}  # 构建目标字典  例如，{0: (313, 1.0), 1: (557, 0.9226), 2: (109, 0.9253)}

        return score_dict

    def ide_tar_items(self, result, target_item_list):
        th_list = []
        for th in self.thresholds:
            count = sum(1 for _, prob in result if prob > th)
            if (len(self.user_representations) * 0.01 < count < len(self.user_representations) * 0.3):
                th_list.append(th)

        top100_ids_list = []
        results_summary = []
        seen_results = set()  # 用于去重
        for n in range(len(th_list)):
            print(f'\n---------------第{n + 1}次阈值计算,th={th_list[n]}---------------')
            th = th_list[n]
            selected_indices = [idx for idx, prob in result if prob > th];
            count = len(set(self.fake_user_ids) & set(selected_indices))
            ratio1 = round(count / len(self.fake_user_ids), 4)
            ratio2 = round(count / len(selected_indices), 4) if len(selected_indices) > 0 else 0.0

            # 当前结果字符串
            key = (ratio1, ratio2)

            # 判断是否重复
            if key not in seen_results:
                seen_results.add(key)
                results_summary.append(f"th={th}时，识别出虚假用户的比例:{ratio1}, 异常样本中虚假用户的占比为:{ratio2}")

                abnormal_samples = self.data[self.data['user'].isin(selected_indices)].copy()
                abnormal_samples.to_csv(f'./datasets/{self.dataset_name}/abnormal_samples.txt',
                                        sep=' ', header=False, index=False)

                iden_tar_items = IdentifyTargets(self.dataset_name, self.num_items, self.lat_fac, self.lr_it,
                                                 self.num_epochs_it, self.lambda_value_it)
                top_target_ids = iden_tar_items.inference_target_items(target_item_list)
                top100_ids_list.append(top_target_ids)

        # print("\n===== 结果汇总 =====")
        # for line in results_summary:
        #     print(line)

        top100_ids_list = top100_ids_list[::-1]
        DR_list = []
        for i in range(2):
            k = len(target_item_list) * i
            top_target_ids_list = [sublist[:k] for sublist in top100_ids_list]

            # 将不同阈值下的计算结果合并到一起，统计出现频次较高的商品为目标商品
            all_items = [item for sublist in top_target_ids_list for item in sublist]
            counter = Counter(all_items)

            positions = defaultdict(list)
            for li, sub in enumerate(top_target_ids_list):
                for pi, item in enumerate(sub):
                    positions[item].append((li, pi))
            pos_multiset = {item: tuple(sorted(pi for _, pi in pairs))
                            for item, pairs in positions.items()}
            earliest_min_list_idx = {}
            for item, pairs in positions.items():
                minpos = min(pi for _, pi in pairs)
                earliest_min_list_idx[item] = min(li for li, pi in pairs if pi == minpos)
            unique_items = list(counter.keys())
            unique_items.sort(key=lambda x: (
                -counter[x],  # 频数高在前
                pos_multiset[x],  # 位置（按升序元组比较）
                earliest_min_list_idx[x],  # 若“位置集合”相同，优先最早达到最小位置的列表
                x  # 最后用 item_id 保证稳定性
            ))
            top_target_ids = unique_items[:k]
            count = len(set(target_item_list) & set(top_target_ids))  # 输出最终识别出的目标商品数量
            DR = round(count / len(target_item_list), 4)
            DR_list.append(DR)

        return DR_list

    # === 判别异常样本
    def method(self, target_item_list, w):
        score_dict = self.cal_abnormal_ratings(w)
        result = [(item_id, prob) for _, (item_id, prob) in score_dict.items()]
        DR_list = self.ide_tar_items(result, target_item_list)

        return DR_list

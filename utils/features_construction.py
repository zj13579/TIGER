import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import pickle

class FeaturesConstruction:
    def __init__(
        self,
        dataset_name: str,
        device: str,
        n_global: int = 256,        # 全局聚类簇数
        global_kmeans_iter: int = 100,
        global_kmeans_batch: int = 2048,
        long_user_refine: bool = False,  # 如需对超长用户再做一次用户内KMeans微调，可设 True
        long_user_thresh: int = 400,     # “超长用户”阈值
        user_kmeans_max_k: int = 7,      # 微调时用户内的最大簇数
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.n_global = int(n_global)
        self.global_kmeans_iter = int(global_kmeans_iter)
        self.global_kmeans_batch = int(global_kmeans_batch)
        self.long_user_refine = bool(long_user_refine)
        self.long_user_thresh = int(long_user_thresh)
        self.user_kmeans_max_k = int(user_kmeans_max_k)

    # =========================
    # 主函数：计算所有特征
    # =========================
    def calculate_features(self, similarity_threshold: float):
        # === 1) 读入数据 ===
        df = pd.read_csv(
            f'./datasets/{self.dataset_name}/combined_dataset.txt',
            sep=' ', header=None, names=['user', 'item', 'rating']
        ).astype({'user': int, 'item': int, 'rating': float})

        # —— numpy 数组（排序+切片）——
        arr_u = df['user'].to_numpy()
        arr_i = df['item'].to_numpy()
        arr_r = df['rating'].to_numpy()
        order = np.argsort(arr_u, kind='mergesort')  # 稳定排序
        arr_u, arr_i, arr_r = arr_u[order], arr_i[order], arr_r[order]
        uniq_u, idx = np.unique(arr_u, return_index=True)
        idx = np.r_[idx, arr_u.size]

        # === 2) item 向量 ===
        item_vectors = torch.load(
            f'./results/{self.dataset_name}/item_vectors.pt', map_location='cpu'
        )
        if isinstance(item_vectors, np.ndarray):
            item_vectors = torch.from_numpy(item_vectors)
        item_vectors = item_vectors.to(self.device)
        item_vectors = F.normalize(item_vectors, dim=1)  # 球面KMeans等价近似

        num_items = int(arr_i.max()) + 1

        # === 3) 统计信息（二维计数表） ===
        # rating 取整→[1..5]→转为 0..4 的 bin
        r_int = np.clip(np.rint(arr_r).astype(int), 1, 5)
        rbin = r_int - 1

        counts = np.zeros((num_items, 5), dtype=np.int32)
        np.add.at(counts, (arr_i, rbin), 1)  # O(N) 构造 (item, rating) 频次表

        item_total = counts.sum(axis=1)  # 每item总评分数
        # item 均值 = Σ(r*freq)/total
        rating_vals = np.arange(1, 6, dtype=np.float64)  # 1..5
        item_sum = (counts * rating_vals).sum(axis=1, dtype=np.float64)
        item_mean = item_sum / np.maximum(item_total, 1)

        # === 4) 全局一次 KMeans（对所有物品） ===
        iv_cpu = item_vectors.detach().cpu().numpy()
        mbk = MiniBatchKMeans(
            n_clusters=self.n_global,
            batch_size=self.global_kmeans_batch,
            n_init=1,
            max_iter=self.global_kmeans_iter,
            random_state=42,
            reassignment_ratio=0.01
        )
        global_labels = mbk.fit_predict(iv_cpu)  # [num_items]
        # 质心单位化（用于可选近似或相似度判断）
        centers = mbk.cluster_centers_.astype(np.float64)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

        # === 5) 输出容器 ===
        user_deviation, user_importance, user_diversity = {}, {}, {}
        user_stability_best, user_stability_avg = {}, {}
        user_rating_count, user_rating_mean, user_rating_var = {}, {}, {}
        user_high_rating_ratio, user_low_rating_ratio = {}, {}

        # 评分权重
        rating_weight = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.2}

        # 便捷引用
        iv = item_vectors  # [num_items, d] (device)
        gl = global_labels  # numpy int array

        # === 6) 遍历每个用户（numpy 切片） ===
        for s, e in zip(idx[:-1], idx[1:]):
            uid = int(arr_u[s])
            item_ids = arr_i[s:e]
            ratings = arr_r[s:e].astype(np.float32)
            m = item_ids.size

            # —— 6.0 基础统计 ——
            user_rating_count[uid] = int(m)
            mean_r = float(np.round(ratings.mean() if m > 0 else 0.0, 4))
            var_r = float(np.round(ratings.var() if m > 0 else 0.0, 4))
            user_rating_mean[uid] = mean_r
            user_rating_var[uid] = var_r

            if m > 0:
                max_rating = float(ratings.max())
                min_rating = float(ratings.min())
                user_high_rating_ratio[uid] = float(np.round((ratings == max_rating).mean(), 4))
                user_low_rating_ratio[uid]  = float(np.round((ratings == min_rating).mean(), 4))
            else:
                user_high_rating_ratio[uid] = 0.0
                user_low_rating_ratio[uid] = 0.0

            if m == 0:
                # 空用户（极少见），全部置零
                user_deviation[uid] = []
                user_importance[uid] = []
                user_diversity[uid] = 0.0
                user_stability_best[uid] = 0.0
                user_stability_avg[uid] = 0.0
                continue

            # —— 6.1 偏离度 & 评分重要性（向量化/计数表 O(1) 查询） ——
            r_bar = item_mean[item_ids]                        # [m]
            dev = np.abs(ratings - r_bar)                      # [m]
            user_deviation[uid] = list(np.round(dev / max(m, 1), 4))

            # (item, rating) 频次 & item 总数
            r_int_u = np.clip(np.rint(ratings).astype(int), 1, 5)
            freq_arr = counts[item_ids, r_int_u - 1].astype(np.int32)
            n_i_arr = item_total[item_ids].astype(np.int32)
            freq_arr = np.maximum(freq_arr, 1)
            n_i_arr = np.maximum(n_i_arr, 1)
            w_arr = np.array([rating_weight.get(int(x), 0.0) for x in r_int_u], dtype=np.float32)
            imp = (1.0 / freq_arr) * np.log(n_i_arr / freq_arr) * w_arr
            user_importance[uid] = list(np.round(imp.astype(np.float32), 4))

            # —— 6.2 偏好多样性（封闭公式，等价于两两平均余弦） ——
            liked_mask = ratings >= 4.0
            L = int(liked_mask.sum())
            if L < 2:
                user_diversity[uid] = 0.0
            else:
                liked_ids = item_ids[liked_mask]
                liked_ids_t = torch.as_tensor(liked_ids, device=self.device, dtype=torch.long)
                liked_vecs = iv.index_select(0, liked_ids_t)  # [L,d] 已单位化
                svec = liked_vecs.sum(dim=0)
                mean_pair_cos = ((svec @ svec) - L) / (L * (L - 1))
                mean_pair_cos = float(mean_pair_cos.detach().cpu().numpy())
                # 数值安全裁剪
                mean_pair_cos = max(-1.0, min(1.0, mean_pair_cos))
                user_diversity[uid] = float(np.round(mean_pair_cos, 4))

            # —— 6.3 偏好稳定性（使用全局簇分桶 + 两个封闭公式） ——
            if m < 2:
                user_stability_best[uid] = 0.0
                user_stability_avg[uid] = 0.0
                continue

            # 用全局簇标签对用户物品分桶
            u_labels = gl[item_ids]  # [m]
            best_sta_k = None
            best_sim = -1.0
            sta_vals = []

            # 逐簇统计
            for k in np.unique(u_labels):
                idx_k = np.where(u_labels == k)[0]
                ck = idx_k.size
                if ck < 2:
                    continue

                cid = item_ids[idx_k]
                # 相似度：封闭公式（向量已单位化）
                cid_t = torch.as_tensor(cid, device=self.device, dtype=torch.long)
                cvec = iv.index_select(0, cid_t)   # [ck, d]
                svec = cvec.sum(dim=0)
                mean_pair_cos = ((svec @ svec) - ck) / (ck * (ck - 1))
                mean_pair_cos = float(mean_pair_cos.detach().cpu().numpy())

                if mean_pair_cos < similarity_threshold:
                    continue

                # 评分稳定性：mean_{i<j}(ri - rj)^2 的封闭公式
                r = ratings[idx_k].astype(np.float64)
                sum_r = r.sum()
                sum_r2 = np.dot(r, r)
                denom = ck * (ck - 1) / 2.0
                sta_k = (ck * sum_r2 - (sum_r ** 2)) / max(denom, 1.0)

                sta_vals.append(sta_k)
                if mean_pair_cos > best_sim:
                    best_sim = mean_pair_cos
                    best_sta_k = sta_k

            # 对超长用户做一次“小范围用户内KMeans微调”
            if (self.long_user_refine and m >= self.long_user_thresh and best_sta_k is None):
                n_clu = int(min(self.user_kmeans_max_k, max(2, int(np.sqrt(m)))))
                ids_t = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
                user_vecs = iv.index_select(0, ids_t).detach().cpu().numpy()
                km = MiniBatchKMeans(
                    n_clusters=n_clu, batch_size=min(256, m),
                    n_init=1, max_iter=50, random_state=42, reassignment_ratio=0.01
                )
                labs = km.fit_predict(user_vecs)
                for kk in range(n_clu):
                    idx_k = np.where(labs == kk)[0]
                    ck = idx_k.size
                    if ck < 2:
                        continue
                    cid = item_ids[idx_k]
                    cid_t = torch.as_tensor(cid, device=self.device, dtype=torch.long)
                    cvec = iv.index_select(0, cid_t)
                    svec = cvec.sum(dim=0)
                    mean_pair_cos = ((svec @ svec) - ck) / (ck * (ck - 1))
                    mean_pair_cos = float(mean_pair_cos.detach().cpu().numpy())
                    if mean_pair_cos < similarity_threshold:
                        continue
                    r = ratings[idx_k].astype(np.float64)
                    sum_r = r.sum(); sum_r2 = np.dot(r, r)
                    denom = ck * (ck - 1) / 2.0
                    sta_k = (ck * sum_r2 - (sum_r ** 2)) / max(denom, 1.0)
                    sta_vals.append(sta_k)
                    if mean_pair_cos > best_sim:
                        best_sim = mean_pair_cos
                        best_sta_k = sta_k

            user_stability_best[uid] = float(np.round(best_sta_k if best_sta_k is not None else 0.0, 4))
            user_stability_avg[uid] = float(np.round(np.mean(sta_vals) if len(sta_vals) > 0 else 0.0, 4))

        # === 4. 示例打印
        print('------------------基础统计特征------------------')
        print("rating_count 示例：", list(user_rating_count.items())[:5])
        print("rating_mean 示例：", list(user_rating_mean.items())[:5])
        print("rating_var 示例：", list(user_rating_var.items())[:5])
        print("high_rating_ratio 示例：", list(user_high_rating_ratio.items())[:5])
        print("low_rating_ratio 示例：", list(user_low_rating_ratio.items())[:5])
        print('------------------构造的特征------------------')
        print("Deviation 示例：", list(user_deviation.items())[:1])
        print("Importance 示例：", list(user_importance.items())[:1])
        print("Diversity 示例：", list(user_diversity.items())[:5])
        print("Stability_best 示例：", list(user_stability_best.items())[:5])
        print("Stability_avg 示例：", list(user_stability_avg.items())[:5])

        # === 5. 保存结果
        combined_data = {
            'r_count': user_rating_count,
            'r_mean': user_rating_mean,
            'r_var': user_rating_var,
            'h_r_ratio': user_high_rating_ratio,
            'l_r_ratio': user_low_rating_ratio,
            'dev': user_deviation,
            'imp': user_importance,
            'div': user_diversity,
            'sta_best': user_stability_best,
            'sta_avg': user_stability_avg,
        }

        # 保存为.pkl 文件
        with open(f'./results/{self.dataset_name}/user_features.pkl', 'wb') as f:
            pickle.dump(combined_data, f)
        print('--------------结果已保存--------------')

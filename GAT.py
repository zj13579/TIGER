import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd
import numpy as np

import random
import os

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.cluster import KMeans
from torch.optim import Adam
import matplotlib.pyplot as plt

from copy import deepcopy

# 设置随机性参数
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 增加确定性设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

class FeatureAligner(nn.Module):
    def __init__(self, item_dim, user_dim):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, user_dim)

    def forward(self, x_user, x_item):
        x_item_proj = self.item_proj(x_item)  # [num_items, user_dim]
        x = torch.cat([x_user, x_item_proj], dim=0)  # [num_users + num_items, user_dim]
        return x

class WeightedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        # 注意力参数做“源/宿”分解，不再拼2F
        self.a_src = nn.Parameter(torch.Tensor(out_channels))
        self.a_dst = nn.Parameter(torch.Tensor(out_channels))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_weight):
        dev = self.lin.weight.device
        x = x.to(dev, dtype=torch.float32).contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.to(dev, dtype=torch.float32)

        # 为保证可复现性，你之前关掉了 TF32/AMP；这里保持一致
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # 线性投影（fp32）
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x = self.lin(x)  # [N, F]

        # 预计算节点打分，内存 O(NF)
        # s_src = <x, a_src>, s_dst = <x, a_dst>
        s_src = (x * self.a_src).sum(dim=-1)  # [N]
        s_dst = (x * self.a_dst).sum(dim=-1)  # [N]

        # 取边两端的打分并相加得到注意力 logits（不构造[E,F]或[E,2F]）
        i, j = edge_index[1], edge_index[0]  # MessagePassing里 index 是目标端 i
        alpha = s_src[j] + s_dst[i]  # [E]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, i)  # 以目标端 i 做归一化

        if edge_weight is not None:
            # 评分权重做乘性调制
            alpha = alpha * edge_weight

        return self.propagate(edge_index, x=x, alpha=alpha)

    def message(self, x_j, alpha):
        # 仅构造 [E, F]（而非 [E, 2F]），显存减半
        return x_j * alpha.unsqueeze(-1)

def message(self, x_i, x_j, edge_weight, index):
    # [E, 2*F] * [1, 2*F] -> [E, 2*F] -> sum(-1) -> [E]
    alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
    alpha = F.leaky_relu(alpha)
    alpha = softmax(alpha, index)  # index 对应目标端 i

    if edge_weight is not None:
        # edge_weight 已是一维连续 [E]
        assert edge_weight.numel() == alpha.numel(), \
            f"edge_weight({edge_weight.numel()}) != alpha({alpha.numel()})"
        alpha = alpha * edge_weight

    return x_j * alpha.unsqueeze(-1)


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gat1 = WeightedGATConv(in_dim, hidden_dim)
        self.gat2 = WeightedGATConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = self.gat2(x, edge_index, edge_attr)
        return x  # 输出节点表示


class LossFunction:
    def __init__(self, item_embs, user_idx, item_idx, alpha):
        # self.user_embs = user_embs
        self.item_embs = item_embs
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.alpha = alpha

    # == (1) 信息丰富度损失
    def predict_ratings_from_embs(self, user_embs):
        u = user_embs[self.user_idx]  # [B, D]
        v = self.item_embs[self.item_idx]  # [B, D]
        r_hat = (u * v).sum(dim=1)  # 内积预测 [B]
        return r_hat

    def info_richness_loss(self, user_embs_gat, user_embs_mf, rating, squared=False):
        # user_idx, item_idx, rating are 1D tensors for a batch
        r_gat = self.predict_ratings_from_embs(user_embs_gat)
        r_mf = self.predict_ratings_from_embs(user_embs_mf)
        if squared:
            e_gat = (r_gat - rating).pow(2)
            e_mf = (r_mf - rating).pow(2)
        else:
            e_gat = torch.abs(r_gat - rating)
            e_mf = torch.abs(r_mf - rating)
        loss_per = F.relu(e_gat - e_mf + self.alpha)  # [B]

        return loss_per.mean()

    # == (2) 聚类清晰度损失
    def generate_triplet_indices(self, user_embeddings, metric='euclidean'):
        N = user_embeddings.size(0)

        if metric == 'cosine':
            norm_emb = F.normalize(user_embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(norm_emb, norm_emb.T)  # [N, N]
        elif metric == 'euclidean':
            dist_matrix = torch.cdist(user_embeddings, user_embeddings, p=2)  # [N, N]
            sim_matrix = -dist_matrix  # 越大越相似

        positive_idx = []
        negative_idx = []

        for i in range(N):
            sim_matrix[i, i] = -float('inf')  # 避免选到自己
            pos = torch.argmax(sim_matrix[i]).item()
            neg = torch.argmin(sim_matrix[i]).item()
            positive_idx.append(pos)
            negative_idx.append(neg)

        return (
            torch.tensor(positive_idx, dtype=torch.long, device=user_embeddings.device),
            torch.tensor(negative_idx, dtype=torch.long, device=user_embeddings.device)
        )

    def cluster_triplet_loss(self, user_embs, positive_idx, negative_idx):
        anchor = user_embs  # shape [N, D]
        positive = user_embs[positive_idx]
        negative = user_embs[negative_idx]

        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        beta = 0.5 * (pos_dist - neg_dist).mean()  # 确定beta的值
        loss = F.relu(pos_dist - neg_dist + beta)

        return loss.mean()


class GATs:
    def __init__(self, n_items, dataset_name, device):
        self.n_items = n_items
        self.dataset_name = dataset_name
        self.df_train = pd.read_csv(f'./datasets/{self.dataset_name}/combined_train_dataset.txt', sep=' ', header=None,
                               names=['user', 'item', 'rating'])
        self.num_users = self.df_train['user'].max() + 1
        self.user_latent_features = torch.load(f'./results/{self.dataset_name}/user_latent_features.pt')
        self.item_vectors = torch.load(f'./results/{self.dataset_name}/item_vectors.pt')  # shape: [num_items, d_item]
        self.device = device

    def prepare_data(self, user_behavior_features, use_latent_features: bool = True):
        # === 0) 先构用户/物品特征 ===
        if use_latent_features:
            user_features = torch.cat([self.user_latent_features, user_behavior_features], dim=1)
        else:
            user_features = user_behavior_features

        # === 1) 以“边里出现的最大ID + 1”为准，统一 U/I/N（与构图严格一致）===
        U_edges = int(self.df_train['user'].max()) + 1
        I_edges = int(self.df_train['item'].max()) + 1
        N_edges = U_edges + I_edges

        # === 2) 将特征矩阵对齐到 U_edges / I_edges（不足则补零，多了则截断）===
        def ensure_rows(mat: torch.Tensor, rows: int) -> torch.Tensor:
            r = int(mat.size(0))
            if r == rows:
                return mat
            if r < rows:
                pad = torch.zeros(rows - r, mat.size(1), dtype=mat.dtype)
                return torch.cat([mat, pad], dim=0)
            else:
                return mat[:rows]

        user_features = ensure_rows(user_features, U_edges)  # [U_edges, du]
        item_vectors = ensure_rows(self.item_vectors, I_edges)  # [I_edges, di]

        # === 3) 构造边（双向），物品统一 +U_edges 偏移 ===
        user_train_indices = torch.tensor(self.df_train['user'].values, dtype=torch.long)
        item_train_indices = torch.tensor(self.df_train['item'].values, dtype=torch.long) + U_edges
        edge_index = torch.stack(
            [torch.cat([user_train_indices, item_train_indices], dim=0),
             torch.cat([item_train_indices, user_train_indices], dim=0)],
            dim=0
        )  # [2, 2E]

        # === 4) 边权重（评分 Min-Max 到 [0,1]，返回一维 [2E]）===
        train_ratings = torch.tensor(self.df_train['rating'].values, dtype=torch.float32)
        rmin, rmax = train_ratings.min(), train_ratings.max()
        train_ratings = (train_ratings - rmin) / (rmax - rmin + 1e-12)
        # edge_attr = torch.cat([train_ratings, train_ratings], dim=0).contiguous()  # [2E]
        edge_attr = torch.cat([train_ratings, train_ratings], dim=0).to(torch.float16).contiguous()

        # === 5) 构建节点特征，行数必须 = N_edges ===
        feature_aligner = FeatureAligner(item_dim=item_vectors.shape[1], user_dim=user_features.shape[1])
        x = feature_aligner(user_features, item_vectors).contiguous()  # 期望 [N_edges, D]
        assert x.size(0) == N_edges, f"x rows {x.size(0)} != N_edges {N_edges} (U={U_edges}, I={I_edges})"
        assert edge_index.max().item() < N_edges and edge_index.min().item() >= 0, \
            f"edge_index out of range: max={edge_index.max().item()} vs N={N_edges}"

        # === 6) 打包 Data 并 detach ===
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.x = data.x.detach()
        data.edge_index = data.edge_index.detach()
        data.edge_attr = data.edge_attr.detach()

        return x, data.x, data.edge_index, data.edge_attr

    def train_model(self, user_behavior_features, num_epochs_gat, hidden_dim,
                    out_dim, lr_gat, alpha, info_th, cluster_th, use_latent_features=True):

        # ========== 准备数据 ==========
        x, data_x, data_edge_index, data_edge_attr = self.prepare_data(
            user_behavior_features, use_latent_features=use_latent_features
        )
        x = x.to(self.device)
        data_x = data_x.to(self.device)
        data_edge_index = data_edge_index.to(self.device)
        data_edge_attr = data_edge_attr.to(self.device)

        # ========== 模型与优化器 ==========
        model = GATEncoder(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim).to(self.device)
        optimizer = Adam(model.parameters(), lr_gat)

        # ========== 测试集（用于 info_richness_loss 里计算误差项）==========
        df_test = pd.read_csv(
            f'./datasets/{self.dataset_name}/combined_test_dataset.txt',
            sep=' ', header=None, names=['user', 'item', 'rating']
        )
        user_test_indices = torch.tensor(df_test['user'].values, dtype=torch.long).to(self.device)
        item_test_indices = torch.tensor(df_test['item'].values, dtype=torch.long).to(self.device)
        test_ratings = torch.tensor(df_test['rating'].values, dtype=torch.float).to(self.device)

        # ========== 损失函数封装 ==========
        loss_function = LossFunction(
            self.item_vectors.to(self.device),
            user_test_indices, item_test_indices,
            alpha
        )

        # ========== 早停与“接近”判定参数 ==========
        close_rel = 0.15  # 相对阈值：|a-b| <= close_rel * ((a+b)/2)
        close_abs = 0.02  # 绝对阈值：|a-b| <= close_abs
        sum_tol = 1e-4  # 最小改进幅度（和的改进需超过该阈值）
        patience = 8  # 在“接近”条件下，和无改进累计到多少轮停止
        min_epochs = 20  # 至少训练这么多轮后才考虑早停

        best_sum = float("inf")
        best_epoch = -1
        best_state = None
        best_user_emb = None
        epochs_since_best = 0

        # 记录曲线：使用两项加权之后的“和”
        loss_list = []

        # 固定当前的权重
        lambda_info = 1.0 / info_th
        lambda_cluster = 1.0 / cluster_th

        print('---------------GAT模型训练开始---------------')
        for epoch in range(num_epochs_gat):
            model.train()
            optimizer.zero_grad()

            node_embeddings = model(data_x, data_edge_index, data_edge_attr)
            user_embeddings = node_embeddings[:self.num_users]

            # --- 两个损失分量 ---
            loss_info = loss_function.info_richness_loss(
                user_embeddings,
                self.user_latent_features.to(self.device),
                test_ratings,
                squared=False
            )
            pos_idx, neg_idx = loss_function.generate_triplet_indices(
                user_embeddings, metric='euclidean'
            )
            loss_cluster = loss_function.cluster_triplet_loss(
                user_embeddings, pos_idx.to(self.device), neg_idx.to(self.device)
            )

            # --- 总损失 ---
            loss = lambda_info * loss_info + lambda_cluster * loss_cluster
            loss.backward()
            optimizer.step()

            # --- 统计、打印与早停判定 ---
            scaled_info = (lambda_info * loss_info).item()
            scaled_clst = (lambda_cluster * loss_cluster).item()
            loss_sum = scaled_info + scaled_clst
            loss_list.append(loss_sum)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, loss: {loss_sum:.6f}, "
                    f"loss_info: {scaled_info:.6f}, loss_cluster: {scaled_clst:.6f}"
                )

            # “接近”判定（绝对或相对满足其一即可）
            diff = abs(scaled_info - scaled_clst)
            mean_mag = 0.5 * (abs(scaled_info) + abs(scaled_clst))
            close = (diff <= close_abs) or (diff <= close_rel * max(mean_mag, 1e-12))

            # 仅在“接近”时，用“和”的最小值更新最佳；并应用 patience
            improved = (loss_sum + sum_tol) < best_sum
            if close and improved and (epoch + 1) >= min_epochs:
                best_sum = loss_sum
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())
                best_user_emb = user_embeddings.detach().cpu()
                epochs_since_best = 0
            elif close:
                epochs_since_best += 1
            # 未接近则不累计耐心

            # 触发早停
            if close and epochs_since_best >= patience:
                print(f"[Early Stop] epoch {epoch + 1}, best (close) sum={best_sum:.6f} @ epoch {best_epoch}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                    user_embeddings = best_user_emb.to(self.device)
                break

        # 若未触发早停但曾有“接近”的最佳，则回滚
        if best_state is not None and best_epoch != -1:
            model.load_state_dict(best_state)
            user_embeddings = best_user_emb.to(self.device)

        # ========== 绘制曲线==========
        plt.figure()
        plt.plot(loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("loss_info + loss_cluster (weighted)")
        plt.title("Training Loss (sum)")
        plt.show()

        # ========== 保存用户表征 ==========
        torch.save(user_embeddings.cpu(), f'./results/{self.dataset_name}/user_representations.pt')
        return user_embeddings
import os, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from attack_models.select_target_items import obtain_target_items


# =========================
# 工具函数
# =========================
def _set_seed(seed: Optional[int]):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _build_dense_matrix(num_users: int, num_items: int, data: pd.DataFrame, device: str):
    mat = torch.zeros((num_users, num_items), dtype=torch.float32, device=device)
    u = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)
    mat[u, i] = r
    return mat

def _choose_selected_items(data: pd.DataFrame, num_items: int,
                           target_item_list: List[int], k: int) -> List[int]:
    # 默认用“全局流行度”挑选 |S| 个 selected items（避开 target）
    counts = data['item'].value_counts()
    popular_items = [int(x) for x in counts.index.tolist() if int(x) not in set(target_item_list)]
    if len(popular_items) < k:
        pool = [i for i in range(num_items) if i not in set(target_item_list)]
        popular_items += pool
    return popular_items[:k]

def _users_with_enough_ratings(user_items: List[List[int]], min_count: int) -> List[int]:
    return [u for u, items in enumerate(user_items) if len(items) >= min_count]

def _build_indices(data: pd.DataFrame, num_users: int):
    # user->(items,ratings) 索引 & item 统计
    user_items = [[] for _ in range(num_users)]
    user_ratings = [[] for _ in range(num_users)]
    for u, i, r in data[['user','item','rating']].itertuples(index=False):
        user_items[u].append(int(i)); user_ratings[u].append(float(r))
    item_mean = data.groupby('item')['rating'].mean().reindex().fillna(3.5)
    item_mean = item_mean.to_dict()
    item_pop = data['item'].value_counts().to_dict()
    return user_items, user_ratings, item_mean, item_pop

def _sample_fillers_for_user(u: int, cfg: AUSHConfig,
                             user_items: List[List[int]],
                             user_ratings: List[List[float]],
                             item_mean: dict, item_pop: dict) -> List[int]:
    pool = user_items[u]
    if len(pool) <= cfg.filler_size: return list(pool)
    if cfg.sampling_strategy == "rating":
        weights = np.array([item_mean.get(it, 3.5) for it in pool], dtype=np.float64)
    elif cfg.sampling_strategy == "pop":
        weights = np.array([item_pop.get(it, 1) for it in pool], dtype=np.float64)
    else:  # "random"
        weights = np.ones(len(pool), dtype=np.float64)
    weights = weights / weights.sum()
    return list(np.random.choice(pool, size=cfg.filler_size, replace=False, p=weights))

def _make_generator(input_dim: int, out_dim: int, cfg: AUSHConfig) -> nn.Module:
    # 塔式 MLP：每层约缩到 1/3，论文 N=5; 这里可自定义
    if cfg.g_hidden is None:
        dims = [input_dim, 400, 133, 44, 14, 4]
        dims = [d for d in dims if d > out_dim] + [out_dim]
    else:
        dims = [input_dim] + cfg.g_hidden + [out_dim]
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.Sigmoid()]
    layers += [nn.Linear(dims[-2], dims[-1]), nn.Sigmoid()]
    return nn.Sequential(*layers)

def _make_discriminator(input_dim: int, cfg: AUSHConfig) -> nn.Module:
    # 简单 MLP 判别器
    h = cfg.d_hidden or [512, 128]
    dims = [input_dim] + h + [1]
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.LeakyReLU(0.2, inplace=True)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)


# =========================
# 核心：AUSH 训练 & 生成
# =========================
def _aush_generate(data: pd.DataFrame,
                   target_item_list: List[int],
                   num_items: int,
                   attack_num: int,
                   cfg: AUSHConfig) -> np.ndarray:
    _set_seed(cfg.seed)
    device = cfg.device

    num_users = int(data['user'].max()) + 1
    # 稠密真实矩阵、索引与统计
    X_real = _build_dense_matrix(num_users, num_items, data, device)   # [U, I]
    user_items, user_ratings, item_mean, item_pop = _build_indices(data, num_users)

    # Selected Items（S）
    selected_items = _choose_selected_items(data, num_items, target_item_list, cfg.num_selected)
    S = torch.tensor(selected_items, dtype=torch.long, device=device)   # [|S|]
    target_tensor = torch.tensor(target_item_list, dtype=torch.long, device=device)

    # 训练样本用户（模板）集合：有足够评分的人
    profile_size = cfg.filler_size + cfg.num_selected + len(target_item_list)
    candidates = _users_with_enough_ratings(user_items, min_count=max(profile_size, 5))
    if len(candidates) == 0:
        raise RuntimeError("No users have enough ratings to serve as templates.")
    # 生成器输入维度 = num_items（把 filler 放在原始索引位）
    G = _make_generator(input_dim=num_items, out_dim=len(selected_items), cfg=cfg).to(device)
    D = _make_discriminator(input_dim=num_items, cfg=cfg).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d)
    bce = nn.BCEWithLogitsLoss()

    # —— 训练 —— #
    steps_per_epoch = max(1, attack_num // max(1, cfg.batch_size))
    for epoch in range(cfg.epochs):
        for _ in range(steps_per_epoch):
            # ------- 判别器 k1 步 -------
            for _ in range(cfg.k_disc):
                # 真实样本
                real_users = np.random.choice(candidates, size=min(cfg.batch_size, len(candidates)), replace=True)
                x_real = X_real[torch.tensor(real_users, dtype=torch.long, device=device)]  # [B, I]
                # 伪样本：按模板 + generator 补丁
                x_fake = torch.zeros_like(x_real)
                for bi, u in enumerate(real_users):
                    fillers = _sample_fillers_for_user(int(u), cfg, user_items, user_ratings, item_mean, item_pop)
                    x_fake[bi, fillers] = X_real[u, fillers]
                # 生成 selected 的预测
                g_in = x_fake.clone()                                # [B, I]
                g_out = G(g_in)                                      # [B, |S|], sigmoid -> (0,1)
                # 映射到评分区间 [1,5]
                g_scores = cfg.rating_min + (cfg.rating_max - cfg.rating_min) * g_out
                x_fake.scatter_(1, S.unsqueeze(0).expand_as(g_scores), g_scores)
                # 目标物品直接设为满分
                x_fake.index_put_((torch.arange(x_fake.size(0), device=device).unsqueeze(1),
                                   target_tensor.unsqueeze(0).expand(x_fake.size(0), -1)),
                                  torch.full((x_fake.size(0), len(target_item_list)),
                                             cfg.rating_max, device=device))
                # 判别器损失
                logits_real = D(x_real).squeeze(-1)
                logits_fake = D(x_fake.detach()).squeeze(-1)
                loss_d = bce(logits_real, torch.ones_like(logits_real)) + \
                         bce(logits_fake, torch.zeros_like(logits_fake))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ------- 生成器 k2 步 -------
            for _ in range(cfg.k_gen):
                real_users = np.random.choice(candidates, size=min(cfg.batch_size, len(candidates)), replace=True)
                x_fake = torch.zeros((len(real_users), num_items), dtype=torch.float32, device=device)
                obs_selected_mask = torch.zeros((len(real_users), len(selected_items)),
                                                dtype=torch.bool, device=device)

                # 组装输入 + 记录哪些 selected 是已观测（用于重构损失）
                for bi, u in enumerate(real_users):
                    fillers = _sample_fillers_for_user(int(u), cfg, user_items, user_ratings, item_mean, item_pop)
                    x_fake[bi, fillers] = X_real[u, fillers]
                    # 标记该用户在 selected 上是否有真实评分
                    obs_selected_mask[bi] = (X_real[u, S] > 0)

                g_in = x_fake.clone()
                g_out = G(g_in)                                      # [B, |S|]
                g_scores = cfg.rating_min + (cfg.rating_max - cfg.rating_min) * g_out

                # 组装最终伪样本（用于对抗损失）
                x_comp = x_fake.clone()
                x_comp.scatter_(1, S.unsqueeze(0).expand_as(g_scores), g_scores)
                x_comp.index_put_((torch.arange(x_comp.size(0), device=device).unsqueeze(1),
                                   target_tensor.unsqueeze(0).expand(x_comp.size(0), -1)),
                                  torch.full((x_comp.size(0), len(target_item_list)),
                                             cfg.rating_max, device=device))

                # 对抗（骗判别器）
                logits_fake = D(x_comp).squeeze(-1)
                loss_adv = bce(logits_fake, torch.ones_like(logits_fake))

                # 重构损失：已观测 selected 用真实评分；未观测按比例抽样当作 0
                recon_targets = torch.zeros_like(g_scores)
                recon_mask = obs_selected_mask.clone()
                # 已观测的 truth
                recon_targets[obs_selected_mask] = X_real[torch.tensor(real_users, dtype=torch.long, device=device)][:, S][obs_selected_mask]
                # 随机采样未观测部分
                if cfg.recon_unobs_ratio > 0.0:
                    unobs_mask = ~obs_selected_mask
                    if unobs_mask.any():
                        # 以 m 的比例从未观测中抽样
                        drop = torch.rand_like(g_scores) > cfg.recon_unobs_ratio
                        sampled_unobs = unobs_mask & (~drop)
                        recon_mask = recon_mask | sampled_unobs  # 未观测采 0 作为重构目标
                loss_recon = F.mse_loss(g_scores[recon_mask], recon_targets[recon_mask]) if recon_mask.any() \
                             else torch.tensor(0.0, device=device)

                # Shilling 损失：把 selected 项也往 Q 推（便于影响 in-segment 用户）
                Q = cfg.rating_max
                loss_shill = F.mse_loss(g_scores, torch.full_like(g_scores, Q))

                loss_g = cfg.lambda_adv * loss_adv + cfg.lambda_recon * loss_recon + cfg.lambda_shill * loss_shill
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    # —— 生成 attack_num 个对抗样本 —— #
    fake_profiles = np.zeros((attack_num, num_items), dtype=np.int16)
    gen_batch = min(cfg.batch_size, attack_num)
    ptr = 0
    while ptr < attack_num:
        B = min(gen_batch, attack_num - ptr)
        real_users = np.random.choice(candidates, size=B, replace=True)
        x_fake = torch.zeros((B, num_items), dtype=torch.float32, device=device)
        for bi, u in enumerate(real_users):
            fillers = _sample_fillers_for_user(int(u), cfg, user_items, user_ratings, item_mean, item_pop)
            x_fake[bi, fillers] = X_real[u, fillers]
        g_scores = cfg.rating_min + (cfg.rating_max - cfg.rating_min) * G(x_fake)
        x_fake.scatter_(1, S.unsqueeze(0).expand_as(g_scores), g_scores)
        x_fake.index_put_((torch.arange(B, device=device).unsqueeze(1),
                           target_tensor.unsqueeze(0).expand(B, -1)),
                          torch.full((B, len(target_item_list)), cfg.rating_max, device=device))
        # 仅对已评分位置离散化到 {1..5}，未评分保持 0
        rated_mask = x_fake > 0
        x_round = torch.where(
            rated_mask,
            x_fake.clamp(cfg.rating_min, cfg.rating_max).round(),
            torch.zeros_like(x_fake)
        )
        x_int = x_round.to(torch.int16).cpu().numpy()
        fake_profiles[ptr:ptr + B] = x_int
        ptr += B
    return fake_profiles

class AttackSimulator:
    """
    一次攻击样本生成（AUSH attack）
      simulator = AttackSimulator(dataset_name, n_items, num_target_items, attack_size)
      target_item_list, fake_profiles = simulator.run(save=True)
    """
    def __init__(self,
                 dataset_name: str = "ml-100k",
                 n_items: int = 1682,
                 num_target_items: int = 5,
                 attack_size: float = 0.05):
        self.dataset_name = dataset_name
        self.n_items = int(n_items)
        self.num_target_items = int(num_target_items)
        self.attack_size = float(attack_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_dir = f"./datasets/{self.dataset_name}"
        self.remained_path = os.path.join(self.dataset_dir, "remained_dataset.txt")

        self.cfg = AUSHConfig(device=str(self.device))

    def run(self, save: bool = True, seed=None) -> Tuple[List[int], np.ndarray]:
        data = pd.read_csv(self.remained_path, sep=' ', header=None, names=['user','item','rating'])

        # 假用户数量
        num_users = int(data['user'].max()) + 1
        attack_num = max(1, math.ceil(self.attack_size * num_users))

        # 目标物品（与前面保持一致）
        target_item_list, _, _, _ = obtain_target_items(self.n_items, data, self.dataset_name, self.num_target_items, seed=seed)

        # 生成
        fake_profiles = _aush_generate(
            data=data,
            target_item_list=target_item_list,
            num_items=self.n_items,
            attack_num=attack_num,
            cfg=self.cfg
        )

        # 保存
        if save:
            os.makedirs(self.dataset_dir, exist_ok=True)
            npy_path = f"./results/{self.dataset_name}/aush_fake_profiles.npy"
            np.save(npy_path, fake_profiles)

        print(f'target_item_list: {list(map(int, target_item_list))}')

        return list(map(int, target_item_list)), fake_profiles
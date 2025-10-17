import os, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from attack_models.select_target_items import obtain_target_items

# =========================
# 工具函数
# =========================
def _pretrain_mf(num_users: int,
                 num_items: int,
                 u_idx: torch.Tensor,
                 i_idx: torch.Tensor,
                 r_val: torch.Tensor,
                 cfg: SGLDConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    device = cfg.device
    U = torch.randn(num_users, cfg.latent_dim, device=device) * 0.01
    V = torch.randn(num_items, cfg.latent_dim, device=device) * 0.01
    U.requires_grad_(True)
    V.requires_grad_(True)
    for _ in range(cfg.pretrain_epochs):
        pred = (U[u_idx] * V[i_idx]).sum(-1)
        loss = F.mse_loss(pred, r_val) + cfg.reg * (U.pow(2).mean() + V.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U, V], create_graph=False)
        with torch.no_grad():
            U -= cfg.pretrain_lr * gU
            V -= cfg.pretrain_lr * gV
    return U.detach(), V.detach()


def _build_observed_mask(num_users: int, num_items: int, data: pd.DataFrame, device: str):
    mask = torch.zeros((num_users, num_items), dtype=torch.bool, device=device)
    ui = torch.tensor(data[['user', 'item']].values, dtype=torch.long, device=device)
    mask[ui[:, 0], ui[:, 1]] = True
    return mask


def _sample_filler_items(popularity: np.ndarray, exclude: set, need: int) -> List[int]:
    """按流行度采样填充物品"""
    probs = popularity.astype(float)
    probs[probs < 1] = 1.0
    probs = probs / probs.sum()
    candidates = np.arange(len(probs))
    chosen = []
    while len(chosen) < need:
        j = int(np.random.choice(candidates, p=probs))
        if j not in exclude:
            exclude.add(j); chosen.append(j)
    return chosen


def _init_fake_entries_and_prior(target_item_list: List[int],
                                 num_items: int,
                                 cfg: SGLDConfig,
                                 data: pd.DataFrame):
    """
    返回：
      f_u, f_i, f_vals（连续变量，后续由 SGLD 更新）
      xi_per_item, inv_var_per_item（每个物品的先验均值与方差逆）
    """
    assert cfg.ratings_per_user >= len(target_item_list), "ratings_per_user 必须 >= 目标物品数"

    # 物品统计：均值/方差 + 流行度
    item_stats = data.groupby('item')['rating'].agg(['mean', 'var']).reindex(range(num_items))
    xi = item_stats['mean'].fillna(3.5).values.astype(np.float32)
    var = item_stats['var'].fillna(1.0).values.astype(np.float32)
    var[var < cfg.prior_var_floor] = cfg.prior_var_floor
    inv_var = 1.0 / var

    pop = np.zeros(num_items, dtype=int)
    vc = data['item'].value_counts()
    pop[vc.index.values] = vc.values

    fake_user_ids, fake_item_ids, fake_values = [], [], []
    tgt_all = [int(x) for x in target_item_list]

    for fu in range(_init_fake_entries_and_prior.attack_num):
        # 每个假用户覆盖的目标子集
        tpu = len(tgt_all) if (cfg.targets_per_user is None) else min(cfg.targets_per_user, len(tgt_all))
        chosen_targets = set(np.random.choice(tgt_all, size=tpu, replace=False).tolist())

        chosen = set(chosen_targets)
        need = max(0, cfg.ratings_per_user - len(chosen))
        if need > 0:
            chosen.update(_sample_filler_items(pop, exclude=set(chosen), need=need))

        for j in sorted(chosen):
            fake_user_ids.append(fu)
            fake_item_ids.append(j)
            # 初始值从先验采样（目标略向高端偏移，提升收敛）
            if j in chosen_targets and np.random.rand() > cfg.soft_target:
                v0 = np.clip(np.random.normal(loc=max(xi[j], 4.5), scale=0.3), cfg.rating_min, cfg.rating_max)
            else:
                v0 = np.clip(np.random.normal(loc=xi[j], scale=np.sqrt(var[j])),
                             cfg.rating_min, cfg.rating_max)
            fake_values.append(float(v0))

    f_u = torch.tensor(fake_user_ids, dtype=torch.long)
    f_i = torch.tensor(fake_item_ids, dtype=torch.long)
    f_vals = torch.tensor(fake_values, dtype=torch.float32, requires_grad=True)
    xi_per_item = torch.tensor(xi, dtype=torch.float32)
    inv_var_per_item = torch.tensor(inv_var, dtype=torch.float32)
    return f_u, f_i, f_vals, xi_per_item, inv_var_per_item
_init_fake_entries_and_prior.attack_num = 0


def _inner_train_few_steps(U0: torch.Tensor, V0: torch.Tensor,
                           num_users: int, attack_num: int,
                           real_u: torch.Tensor, real_i: torch.Tensor, real_r: torch.Tensor,
                           f_u: torch.Tensor, f_i: torch.Tensor, f_r: torch.Tensor,
                           cfg: SGLDConfig):
    """对 (U,V) 做几步GD，保留计算图"""
    device = cfg.device
    U = torch.zeros((num_users + attack_num, cfg.latent_dim), device=device)
    U[:num_users] = U0
    V = V0.clone()

    U.requires_grad_(True)
    V.requires_grad_(True)

    f_u_expanded = f_u + num_users
    all_u = torch.cat([real_u, f_u_expanded], dim=0)
    all_i = torch.cat([real_i, f_i], dim=0)
    all_r = torch.cat([real_r, f_r], dim=0)

    for _ in range(cfg.inner_steps):
        pred = (U[all_u] * V[all_i]).sum(-1)
        loss = F.mse_loss(pred, all_r) + cfg.reg * (U.pow(2).mean() + V.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U, V], create_graph=True)
        U = U - cfg.inner_lr * gU
        V = V - cfg.inner_lr * gV
    return U, V


def _push_objective(U: torch.Tensor, V: torch.Tensor,
                    target_item_list: List[int],
                    observed_mask: torch.Tensor,
                    num_users: int) -> torch.Tensor:
    """提升真实用户对目标物品的预测分数（仅未评分处）"""
    target_idx = torch.tensor(target_item_list, dtype=torch.long, device=U.device)
    U_real = U[:num_users]
    V_t = V[target_idx]
    scores = U_real @ V_t.T
    submask = (~observed_mask[:, target_idx])
    return scores[submask].mean() if submask.any() else scores.mean()


def _sgld_generate(data: pd.DataFrame,
                   target_item_list: List[int],
                   num_items: int,
                   attack_num: int,
                   cfg: SGLDConfig) -> np.ndarray:
    """核心：SGLD 采样生成 fake_profiles (attack_num x num_items)"""
    device = cfg.device
    num_users = int(data['user'].max()) + 1

    # 真实数据张量
    u_idx = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i_idx = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r_val = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)

    # 1) 预训练 MF
    U0, V0 = _pretrain_mf(num_users, num_items, u_idx, i_idx, r_val, cfg)

    # 2) 假评分变量 + 先验统计
    _init_fake_entries_and_prior.attack_num = attack_num
    f_u, f_i, f_vals, xi_per_item, inv_var_per_item = _init_fake_entries_and_prior(
        target_item_list, num_items, cfg, data
    )
    f_u = f_u.to(device)
    f_i = f_i.to(device)
    f_vals = f_vals.to(device); f_vals.requires_grad_(True)
    xi_per_item = xi_per_item.to(device)
    inv_var_per_item = inv_var_per_item.to(device)

    # 3) 掩码
    observed_mask = _build_observed_mask(num_users, num_items, data, device)

    # 4) SGLD 外层
    step = float(cfg.sgld_step)
    sqrt_step = math.sqrt(step)
    for _ in range(cfg.sgld_iters):
        # few-step 更新 U,V
        U, V = _inner_train_few_steps(U0, V0, num_users, attack_num,
                                      u_idx, i_idx, r_val,
                                      f_u, f_i, f_vals, cfg)

        # 目标项梯度（攻方效用）
        obj = _push_objective(U, V, target_item_list, observed_mask, num_users)
        if f_vals.grad is not None:
            f_vals.grad.zero_()
        grad_like = torch.autograd.grad(obj, f_vals, retain_graph=False)[0]  # dR/d(tildeM)

        # 先验项梯度：-(f_vals - xi_j)/sigma_j^2
        xi_j = xi_per_item[f_i]
        invvar_j = inv_var_per_item[f_i]
        grad_prior = -(f_vals - xi_j) * invvar_j

        # SGLD 更新：x_{t+1} = x_t + (ε/2)(grad_prior + β*grad_like) + sqrt(ε)*N(0,I)
        noise = torch.randn_like(f_vals) * sqrt_step
        with torch.no_grad():
            f_vals += 0.5 * step * (grad_prior + cfg.beta * grad_like) + noise
            f_vals.clamp_(cfg.rating_min, cfg.rating_max)

    # 5) 稠密矩阵（结尾离散化，保持与真实数据一致）
    with torch.no_grad():
        f_vals_disc = f_vals.clamp(cfg.rating_min, cfg.rating_max).round()

    fake_profiles = np.zeros((attack_num, num_items), dtype=np.int16)
    for uid, iid, val in zip(f_u.cpu().numpy(), f_i.cpu().numpy(),
                             f_vals_disc.detach().cpu().numpy().astype(np.int16)):
        fake_profiles[int(uid), int(iid)] = int(val)
    return fake_profiles


class AttackSimulator:
    """
    一次攻击样本生成（SGLD attack）
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

        self.cfg = SGLDConfig(device=str(self.device))

    def run(self, save: bool = True, seed=None) -> Tuple[List[int], np.ndarray]:
        # 读数据
        data = pd.read_csv(self.remained_path, sep=' ', header=None, names=['user', 'item', 'rating'])

        # 假用户数量
        num_users = int(data['user'].max()) + 1
        attack_num = max(1, math.ceil(self.attack_size * num_users))

        # 目标物品
        target_item_list, _, _, _ = obtain_target_items(self.n_items, data, self.dataset_name, self.num_target_items, seed=seed)

        # 支持集最小覆盖
        self.cfg.ratings_per_user = max(self.cfg.ratings_per_user, len(target_item_list))

        # 生成
        fake_profiles = _sgld_generate(
            data=data,
            target_item_list=target_item_list,
            num_items=self.n_items,
            attack_num=attack_num,
            cfg=self.cfg
        )

        # 保存
        if save:
            os.makedirs(self.dataset_dir, exist_ok=True)
            npy_path = f"./results/{self.dataset_name}/sgld_fake_profiles.npy"
            np.save(npy_path, fake_profiles)

        print(f'target_item_list: {list(map(int, target_item_list))}')

        return list(map(int, target_item_list)), fake_profiles


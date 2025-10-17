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
                 cfg: PGAConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """仅用真实数据做一个简洁 MF 预训练，返回 U,V（不保留图）"""
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


def _sample_filler_items(popularity: np.ndarray, exclude: set, need: int, rng: np.random.Generator) -> List[int]:
    """按流行度采样填充物品，提高拟态性"""
    probs = popularity.copy().astype(float)
    probs[probs < 1] = 1.0
    probs = probs / probs.sum()
    candidates = np.arange(len(probs))
    chosen = []
    while len(chosen) < need:
        j = int(rng.choice(candidates, p=probs))
        if j not in exclude:
            exclude.add(j); chosen.append(j)
    return chosen


def _init_fake_entries(target_item_list: List[int],
                       num_items: int,
                       cfg: PGAConfig,
                       data: pd.DataFrame,
                       seed_for_init: int = 0):
    """
    固定支持集（B个条目）：目标子集 + 填充物品；
    返回：f_u, f_i, f_vals（连续变量，后续由 PGA 更新）
    """
    assert cfg.ratings_per_user >= len(target_item_list), "ratings_per_user 必须 >= 目标物品数"
    rng = np.random.default_rng(seed_for_init)

    # 物品流行度与经验均值
    pop = np.zeros(num_items, dtype=int)
    cnt = data['item'].value_counts()
    pop[cnt.index.values] = cnt.values
    item_mean = data.groupby('item')['rating'].mean().reindex(range(num_items)).fillna(3.5).values

    fake_user_ids, fake_item_ids, fake_values = [], [], []
    tgt_all = [int(x) for x in target_item_list]

    for fu in range(_init_fake_entries.attack_num):
        # 每个假用户覆盖的目标子集
        tpu = len(tgt_all) if (cfg.targets_per_user is None) else min(cfg.targets_per_user, len(tgt_all))
        chosen_targets = set(rng.choice(tgt_all, size=tpu, replace=False).tolist())

        # 支持集 = 目标子集 + 填充物品
        chosen = set(chosen_targets)
        need = max(0, cfg.ratings_per_user - len(chosen))
        if need > 0:
            chosen.update(_sample_filler_items(pop, exclude=set(chosen), need=need, rng=rng))

        for j in sorted(chosen):
            fake_user_ids.append(fu)
            fake_item_ids.append(j)
            if j in chosen_targets:
                # 目标条目：多数给满分，少部分给接近满分（更拟态）
                if rng.random() < cfg.soft_target:
                    v0 = float(np.clip(rng.normal(loc=4.6, scale=0.3), cfg.rating_min, cfg.rating_max))
                else:
                    v0 = cfg.rating_max
            else:
                v0 = float(item_mean[j])
                v0 = min(cfg.rating_max, max(cfg.rating_min, v0))
            fake_values.append(v0)

    f_u = torch.tensor(fake_user_ids, dtype=torch.long)
    f_i = torch.tensor(fake_item_ids, dtype=torch.long)
    f_vals = torch.tensor(fake_values, dtype=torch.float32, requires_grad=True)
    return f_u, f_i, f_vals
_init_fake_entries.attack_num = 0


def _inner_train_few_steps(U0: torch.Tensor, V0: torch.Tensor,
                           num_users: int, attack_num: int,
                           real_u: torch.Tensor, real_i: torch.Tensor, real_r: torch.Tensor,
                           f_u: torch.Tensor, f_i: torch.Tensor, f_r: torch.Tensor,
                           cfg: PGAConfig):
    """对 (U,V) 做几步 GD，保留计算图"""
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
    """提升真实用户对目标物品的预测分数（仅在未评分处求平均）"""
    target_idx = torch.tensor(target_item_list, dtype=torch.long, device=U.device)
    U_real = U[:num_users]
    V_t = V[target_idx]
    scores = U_real @ V_t.T
    submask = (~observed_mask[:, target_idx])
    return scores[submask].mean() if submask.any() else scores.mean()


def _pga_generate(data: pd.DataFrame,
                  target_item_list: List[int],
                  num_items: int,
                  attack_num: int,
                  cfg: PGAConfig) -> np.ndarray:
    """PGA push attack 生成 fake_profiles (attack_num x num_items)"""
    device = cfg.device
    num_users = int(data['user'].max()) + 1

    # 真实数据张量
    u_idx = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i_idx = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r_val = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)

    # 1) 预训练 MF
    U0, V0 = _pretrain_mf(num_users, num_items, u_idx, i_idx, r_val, cfg)

    # 2) 假评分变量（固定支持集）
    _init_fake_entries.attack_num = attack_num
    f_u, f_i, f_vals = _init_fake_entries(target_item_list, num_items, cfg, data)
    f_u = f_u.to(device)
    f_i = f_i.to(device)
    f_vals = f_vals.to(device); f_vals.requires_grad_(True)

    # 3) 真实用户是否已评分掩码
    observed_mask = _build_observed_mask(num_users, num_items, data, device)

    # 4) PGA 外层（只做区间裁剪，不强制满分）
    tgt_tensor = torch.tensor(target_item_list, dtype=torch.long, device=device)
    for _ in range(cfg.pga_iters):
        with torch.no_grad():
            f_vals.clamp_(cfg.rating_min, cfg.rating_max)

        # 内层 few-step 更新 U,V
        U, V = _inner_train_few_steps(U0, V0, num_users, attack_num,
                                      u_idx, i_idx, r_val,
                                      f_u, f_i, f_vals, cfg)

        # 目标：提升真实用户对目标物品的平均预测分
        obj = _push_objective(U, V, target_item_list, observed_mask, num_users)

        # PGA 上升一步 + 投影
        if f_vals.grad is not None:
            f_vals.grad.zero_()
        grad = torch.autograd.grad(obj, f_vals, retain_graph=False)[0]
        with torch.no_grad():
            f_vals += cfg.pga_step * grad
            f_vals.clamp_(cfg.rating_min, cfg.rating_max)

    # 5) 落盘前离散化（保证与真实数据一致）
    with torch.no_grad():
        f_vals_disc = f_vals.clamp(cfg.rating_min, cfg.rating_max).round()

    fake_profiles = np.zeros((attack_num, num_items), dtype=np.int16)
    for uid, iid, val in zip(f_u.cpu().numpy(), f_i.cpu().numpy(),
                             f_vals_disc.detach().cpu().numpy().astype(np.int16)):
        fake_profiles[int(uid), int(iid)] = int(val)
    return fake_profiles


class AttackSimulator:
    """
    一次攻击样本生成（PGA push attack）
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

        # 数据路径（与现有目录约定保持一致）
        self.dataset_dir = f"./datasets/{self.dataset_name}"
        self.remained_path = os.path.join(self.dataset_dir, "remained_dataset.txt")

        # 默认配置（可在外部修改 self.cfg 的属性）
        self.cfg = PGAConfig(device=str(self.device))

    def run(self, save: bool = True, seed=None) -> Tuple[List[int], np.ndarray]:
        # 读取数据
        data = pd.read_csv(self.remained_path, sep=' ', header=None, names=['user', 'item', 'rating'])

        # 假用户数量：ceil(attack_size * #users)，至少 1
        num_users = int(data['user'].max()) + 1
        attack_num = max(1, math.ceil(self.attack_size * num_users))
        target_item_list, _, _, _ = obtain_target_items(self.n_items, data, self.dataset_name, self.num_target_items, seed=seed)

        # 每个假用户的支持集至少覆盖全部目标
        self.cfg.ratings_per_user = max(self.cfg.ratings_per_user, len(target_item_list))

        # 生成对抗样本
        fake_profiles = _pga_generate(
            data=data,
            target_item_list=target_item_list,
            num_items=self.n_items,
            attack_num=attack_num,
            cfg=self.cfg
        )

        # 保存：.npy 与 .txt（空格分隔、无表头）
        if save:
            os.makedirs(self.dataset_dir, exist_ok=True)
            npy_path = f"./results/{self.dataset_name}/pga_fake_profiles.npy"
            np.save(npy_path, fake_profiles)

        print(f'target_item_list: {list(map(int, target_item_list))}')

        return list(map(int, target_item_list)), fake_profiles

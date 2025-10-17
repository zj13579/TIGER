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

def _dense(num_users: int, num_items: int, data: pd.DataFrame, device: str):
    X = torch.zeros((num_users, num_items), dtype=torch.float32, device=device)
    u = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)
    X[u, i] = r
    return X

def _build_stats(data: pd.DataFrame, num_users: int, num_items: int):
    user_items = [[] for _ in range(num_users)]
    for u, i, r in data[['user','item','rating']].itertuples(index=False):
        user_items[u].append(int(i))
    item_mean = data.groupby('item')['rating'].mean().reindex(range(num_items)).fillna(3.5).values
    item_pop  = data['item'].value_counts().reindex(range(num_items)).fillna(0).values

    # 每个物品的评分直方图（1..5）
    hist = np.zeros((num_items, 5), dtype=np.float64)
    for i, r in data[['item','rating']].itertuples(index=False):
        hist[int(i), int(r)-1] += 1
    hist_sum = hist.sum(axis=1, keepdims=True); hist_sum[hist_sum==0]=1
    hist_prob = hist / hist_sum
    return user_items, item_mean, item_pop, hist_prob

def _pretrain_mf(num_users, num_items, u_idx, i_idx, r_val, cfg: TrialCfg):
    device = cfg.device
    U = torch.randn(num_users, cfg.latent_dim, device=device) * 0.01
    V = torch.randn(num_items, cfg.latent_dim, device=device) * 0.01
    U.requires_grad_(True); V.requires_grad_(True)
    for _ in range(10):
        pred = (U[u_idx] * V[i_idx]).sum(-1)
        loss = F.mse_loss(pred, r_val) + cfg.reg*(U.pow(2).mean()+V.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U, V], create_graph=False, allow_unused=True)
        if gU is None: gU = torch.zeros_like(U)
        if gV is None: gV = torch.zeros_like(V)
        with torch.no_grad():
            U -= cfg.inner_lr * gU
            V -= cfg.inner_lr * gV
    return U.detach(), V.detach()

def _inner_mf_few_steps(U0, V0, real_u, real_i, real_r, fake_dense, cfg: TrialCfg):
    """
    保持可微：返回 (U,V)，它们对 fake_dense 有依赖，从而
    attack_loss(U,V) 对 fake_dense、进而对 G(·) 有梯度。
    """
    device = cfg.device
    num_users = U0.size(0)
    B = fake_dense.size(0)
    U = torch.zeros((num_users + B, cfg.latent_dim), device=device)
    U[:num_users] = U0
    V = V0.clone()
    U.requires_grad_(True); V.requires_grad_(True)

    # 稀疏化 batch 假用户
    fu, fi, fr = [], [], []
    for b in range(B):
        idx = torch.nonzero(fake_dense[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel()==0: continue
        fu.append(torch.full((idx.numel(),), num_users+b, dtype=torch.long, device=device))
        fi.append(idx); fr.append(fake_dense[b, idx])
    if len(fu)>0:
        fu = torch.cat(fu); fi = torch.cat(fi); fr = torch.cat(fr)
        all_u = torch.cat([real_u, fu], 0)
        all_i = torch.cat([real_i, fi], 0)
        all_r = torch.cat([real_r, fr], 0)
    else:
        # 极端情况兜底：若这一批没有任何伪条目，把一个极小“dummy”与 fake_dense 相连，避免断图
        all_u, all_i, all_r = real_u, real_i, real_r + (fake_dense.sum()*0)

    for _ in range(cfg.inner_steps):
        pred = (U[all_u] * V[all_i]).sum(-1)
        loss = F.mse_loss(pred, all_r) + cfg.reg*(U.pow(2).mean()+V.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U,V], create_graph=True, allow_unused=True, retain_graph=True)
        if gU is None: gU = torch.zeros_like(U)
        if gV is None: gV = torch.zeros_like(V)
        U = U - cfg.inner_lr * gU
        V = V - cfg.inner_lr * gV
    return U, V

def _attack_loss(U, V, target_idx: torch.Tensor, cfg: TrialCfg, sample_users: Optional[torch.Tensor]=None):
    # 推高 target 的 softmax 概率（负对数似然）
    if sample_users is None:
        num_users = U.size(0)
        m = min(cfg.atk_eval_users, num_users)
        idx = np.random.choice(num_users, size=m, replace=False)
        sample_users = torch.tensor(idx, dtype=torch.long, device=U.device)
    S = U[sample_users] @ V.T  # [B, I]
    logp = torch.log_softmax(S, dim=1)
    # 针对多个 target 取平均
    loss = -logp[:, target_idx].mean()
    return loss


# =========================
# 生成器
# =========================
class Generator(nn.Module):
    """输入 e（0 表示未选条目，>0 表示被采样的条目且给一个“参考值”），输出假评分"""
    def __init__(self, num_items: int, cfg: TrialCfg):
        super().__init__()
        h = cfg.g_hidden or [512, 256, 128]
        dims = [num_items] + h + [num_items]
        layers = []
        for a,b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a,b), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1]), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(self, e):
        # 仅对 e>0 的位置生成评分，其它保持 0
        mask = (e > 0).float()
        g = self.net(e)  # (0,1)
        vals = g*(self.cfg.rating_max - self.cfg.rating_min) + self.cfg.rating_min
        return vals * mask


# =========================
# 噪声采样器（从真实分布构造模板 e）
# =========================
class NoiseSampler:
    def __init__(self, X_real: torch.Tensor, user_items: List[List[int]],
                 item_mean: np.ndarray, item_pop: np.ndarray, hist_prob: np.ndarray,
                 cfg: TrialCfg):
        self.X_real = X_real
        self.user_items = user_items
        self.item_mean = item_mean
        self.item_pop  = item_pop
        self.hist_prob = hist_prob
        self.cfg = cfg

    def sample(self, B: int, num_items: int, target_idx: List[int]) -> torch.Tensor:
        device = self.X_real.device
        e = torch.zeros((B, num_items), dtype=torch.float32, device=device)
        for b in range(B):
            # 预算：profile_size，必须覆盖所有 target
            k = max(len(target_idx), self.cfg.profile_size)
            if self.cfg.sampling_strategy == "rating":
                probs = self.item_mean / (self.item_mean.sum() + 1e-6)
            elif self.cfg.sampling_strategy == "pop":
                probs = self.item_pop / (self.item_pop.sum() + 1e-6)
            else:
                probs = np.ones(num_items, dtype=np.float64) / num_items

            chosen = set(target_idx)
            while len(chosen) < k:
                j = int(np.random.choice(num_items, p=probs))
                chosen.add(j)
            chosen = sorted(list(chosen))
            # 评分值按直方图采样（1..5），作为“参考值”提供给 G 的输入
            for j in chosen:
                p = self.hist_prob[j]
                r = int(np.argmax(np.random.multinomial(1, p))) + 1
                e[b, j] = float(r)
        return e


# =========================
# 主流程（极简 TrialAttack）
# =========================
def _trialattack_generate(data: pd.DataFrame,
                          target_item_list: List[int],
                          num_items: int,
                          attack_num: int,
                          cfg: TrialCfg) -> np.ndarray:
    _set_seed(cfg.seed)
    device = cfg.device
    num_users = int(data['user'].max()) + 1

    # 真实矩阵/索引/统计
    X_real = _dense(num_users, num_items, data, device)
    user_items, item_mean, item_pop, hist_prob = _build_stats(data, num_users, num_items)

    # 预训练 MF（代理）
    u_idx = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i_idx = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r_val = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)
    U0, V0 = _pretrain_mf(num_users, num_items, u_idx, i_idx, r_val, cfg)

    sampler = NoiseSampler(X_real, user_items, item_mean, item_pop, hist_prob, cfg)
    target_tensor = torch.tensor(target_item_list, dtype=torch.long, device=device)

    # 模型
    G = Generator(num_items, cfg).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g)

    steps_per_epoch = max(1, attack_num // max(1, cfg.batch_size))

    # ===== 训练 G（单模块，可微）=====
    for _ in range(cfg.epochs):
        for _s in range(steps_per_epoch):
            B = cfg.batch_size
            e = sampler.sample(B, num_items, target_item_list)  # [B,I]（0 表未选）
            x_fake = G(e)                                       # [B,I] 连续

            # push attack：目标物品设为满分（保持对 G 可微）
            row_idx = torch.arange(B, device=device).unsqueeze(1)
            x_fake.index_put_((row_idx, target_tensor.unsqueeze(0).expand(B, -1)),
                              torch.full((B, len(target_item_list)), cfg.rating_max, device=device))

            # few-step MF（可微），得到 U,V
            U, V = _inner_mf_few_steps(U0, V0, u_idx, i_idx, r_val, x_fake, cfg)

            # 攻击损失（推高 target 排名）
            loss_direct = _attack_loss(U, V, target_tensor, cfg)

            # 拟态损失：对已选条目（e>0）让 x_fake 接近 e（保持隐蔽）
            mask = (e > 0)
            loss_recon = F.mse_loss(x_fake[mask], e[mask]) if mask.any() else (x_fake*0).sum()

            # 轻微 L2
            loss_l2 = sum((p.pow(2).sum() for p in G.parameters())) * cfg.lambda_l2

            loss = cfg.lambda_direct*loss_direct + cfg.lambda_recon*loss_recon + loss_l2

            opt_g.zero_grad()
            loss.backward()
            opt_g.step()

    # ===== 生成 attack_num 个假用户 & 离散化 =====
    fake_profiles = np.zeros((attack_num, num_items), dtype=np.int16)
    ptr, Bgen = 0, min(cfg.batch_size, attack_num)
    while ptr < attack_num:
        B = min(Bgen, attack_num - ptr)
        e = sampler.sample(B, num_items, target_item_list)
        x_fake = G(e)
        row_idx = torch.arange(B, device=device).unsqueeze(1)
        x_fake.index_put_((row_idx, target_tensor.unsqueeze(0).expand(B, -1)),
                          torch.full((B, len(target_item_list)), cfg.rating_max, device=device))
        # 仅对 >0 的位置离散到 {1..5}，未评分保持 0
        rated = x_fake > 0
        x_round = torch.where(rated,
                              x_fake.clamp(cfg.rating_min, cfg.rating_max).round(),
                              torch.zeros_like(x_fake))
        fake_profiles[ptr:ptr+B] = x_round.to(torch.int16).cpu().numpy()
        ptr += B
    return fake_profiles


class AttackSimulator:
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
        self.cfg = TrialCfg(device=str(self.device))

    def run(self, save: bool = True, seed=None) -> Tuple[List[int], np.ndarray]:
        data = pd.read_csv(self.remained_path, sep=' ', header=None, names=['user','item','rating'])
        num_users = int(data['user'].max()) + 1
        attack_num = max(1, math.ceil(self.attack_size * num_users))

        target_item_list, _, _, _ = obtain_target_items(self.n_items, data, self.dataset_name, self.num_target_items, seed=seed)

        if self.cfg.profile_size <= 0:
            avg = int(data.groupby('user').size().mean())
            self.cfg.profile_size = max(avg, len(target_item_list))

        fake_profiles = _trialattack_generate(
            data=data,
            target_item_list=target_item_list,
            num_items=self.n_items,
            attack_num=attack_num,
            cfg=self.cfg
        )

        if save:
            os.makedirs(self.dataset_dir, exist_ok=True)
            npy_path = f"./results/{self.dataset_name}/trialattack_fake_profiles.npy"
            np.save(npy_path, fake_profiles)

        print(f'target_item_list: {list(map(int, target_item_list))}')

        return list(map(int, target_item_list)), fake_profiles

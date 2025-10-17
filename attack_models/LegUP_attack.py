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
# 工具
# =========================
def _set_seed(seed: Optional[int]):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _dense_matrix(num_users: int, num_items: int, data: pd.DataFrame, device: str):
    X = torch.zeros((num_users, num_items), dtype=torch.float32, device=device)
    u = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)
    X[u, i] = r
    return X

def _build_indices(data: pd.DataFrame, num_users: int):
    user_items = [[] for _ in range(num_users)]
    user_ratings = [[] for _ in range(num_users)]
    for u, i, r in data[['user','item','rating']].itertuples(index=False):
        user_items[u].append(int(i)); user_ratings[u].append(float(r))
    item_mean = data.groupby('item')['rating'].mean().fillna(3.5).to_dict()
    item_pop  = data['item'].value_counts().to_dict()
    return user_items, user_ratings, item_mean, item_pop

def _users_with_at_least(user_items: List[List[int]], k: int) -> List[int]:
    return [u for u, items in enumerate(user_items) if len(items) >= k]

def _sample_fillers(u: int, cfg: LegUPConfig,
                    user_items, user_ratings, item_mean, item_pop, need: int) -> List[int]:
    pool = user_items[u]
    if len(pool) <= need: return list(pool)
    if cfg.sampling_strategy == "rating":
        w = np.array([item_mean.get(it, 3.5) for it in pool], dtype=np.float64)
    elif cfg.sampling_strategy == "pop":
        w = np.array([item_pop.get(it, 1) for it in pool], dtype=np.float64)
    else:
        w = np.ones(len(pool), dtype=np.float64)
    w = w / w.sum()
    return list(np.random.choice(pool, size=need, replace=False, p=w))

# Heaviside 近似（式(16)的三段线性近似）
def _Hprime(x: torch.Tensor, tau: torch.Tensor, xi: float) -> torch.Tensor:
    """
    x:   [B, S, 1]  in [0,1]
    tau: [B, S, 5]  cumulative thresholds in (0,1]
    return: [B, S, 5] piecewise-linear values in [0,1]
    """
    # 统一到同一形状做逐段计算
    x = x.expand_as(tau)                        # [B,S,5]
    taum = torch.minimum(tau, 1.0 - tau)        # [B,S,5]

    left  = x < (tau - taum / 2)                # [B,S,5]
    right = x > (tau + taum / 2)                # [B,S,5]
    mid   = ~(left | right)

    out = torch.zeros_like(tau)                 # [B,S,5]
    out = torch.where(left,
                      x * xi / (tau - taum / 2 + 1e-8),
                      out)
    out = torch.where(right,
                      (x - tau) * (1 - 2 * xi) / (taum + 1e-8) + 0.5,
                      out)
    out = torch.where(mid, torch.full_like(tau, 0.5), out)
    return out.clamp(0.0, 1.0)


# =========================
# 生成器与判别器
# =========================
class Generator(nn.Module):
    """
    输入：x_in [B, I]（模板填充后的向量，selected 位置暂空）
    输出：对 selected 的离散评分（1..5）的期望，shape [B, |S|]，且反传可微
    """
    def __init__(self, num_items: int, num_selected: int, cfg: LegUPConfig):
        super().__init__()
        self.cfg = cfg
        D = num_items
        H = cfg.enc_dim
        self.enc = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )
        # 生成偏好（0..1）
        self.pref_head = nn.Sequential(
            nn.Linear(H, num_selected),
            nn.Sigmoid()
        )
        # 产生离散化阈值段（5 段，和=1），个性化阈值由样本隐表征产生
        self.tau_head = nn.Linear(H, 5)

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          rating_disc: [B, |S|]，离散化后的(1..5)期望（连续可微，最后落盘再取整）
          pref: [B, |S|] in (0,1)
          tau_cum: [B, 5] 每个样本的累计阈值（单调递增，和=1）
        """
        h = self.enc(x_in)
        pref = self.pref_head(h)  # (0,1)
        tau_logits = self.tau_head(h)                # [B, 5]
        tau_seg = torch.softmax(tau_logits, dim=-1)  # 每段比例
        tau_cum = torch.cumsum(tau_seg, dim=-1)      # [B,5], 单调递增∈(0,1]

        # sum_k H'(pref, cumsum_k(tau))  →  0..5
        B, S = pref.shape
        # 扩维对齐：[B,S,1] vs [B,1,5]
        x = pref.unsqueeze(-1)                       # [B,S,1]
        tau = tau_cum.unsqueeze(1).expand(B, S, 5)   # [B,S,5]
        Hsum = _Hprime(x, tau, self.cfg.xi).sum(dim=-1)  # [B,S] in [0,5]
        rating_disc = Hsum + 0.0  # (0..5)，后续映射到 {1..5}
        # 按论文式(4)(6)，1..5 = Hsum 本身；这里保留连续，落盘前再离散
        rating_disc = rating_disc  # 已是 0..5 的计数；下游将 +1*mask 处理
        return rating_disc, pref, tau_cum


class Discriminator(nn.Module):
    def __init__(self, num_items: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_items, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),       nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


# =========================
# 代理模型（few-step MF，可微）
# =========================
def _inner_mf_few_steps(U0: torch.Tensor, V0: torch.Tensor,
                        real_u: torch.Tensor, real_i: torch.Tensor, real_r: torch.Tensor,
                        fake_dense: torch.Tensor,         # [B, I]
                        cfg: LegUPConfig):
    """
    把 batch 假用户拼到 U 后端，做 few-step 显式评分 MF（MSE+L2），保留图。
    """
    device = cfg.device
    num_users = U0.size(0)
    num_items = V0.size(0)
    B = fake_dense.size(0)

    U = torch.zeros((num_users + B, cfg.latent_dim), device=device)
    U[:num_users] = U0
    V = V0.clone()
    U.requires_grad_(True); V.requires_grad_(True)

    # 稀疏化 batch 假用户
    fu, fi, fr = [], [], []
    for b in range(B):
        idx = torch.nonzero(fake_dense[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() == 0: continue
        fu.append(torch.full((idx.numel(),), num_users + b, dtype=torch.long, device=device))
        fi.append(idx)
        fr.append(fake_dense[b, idx])
    if len(fu) > 0:
        fu = torch.cat(fu); fi = torch.cat(fi); fr = torch.cat(fr)
        all_u = torch.cat([real_u, fu], dim=0)
        all_i = torch.cat([real_i, fi], dim=0)
        all_r = torch.cat([real_r, fr], dim=0)
    else:
        all_u, all_i, all_r = real_u, real_i, real_r

    for _ in range(cfg.inner_steps):
        pred = (U[all_u] * V[all_i]).sum(-1)
        loss = F.mse_loss(pred, all_r) + cfg.reg * (U.pow(2).mean() + V.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U, V], create_graph=True)
        U = U - cfg.inner_lr * gU
        V = V - cfg.inner_lr * gV
    return U, V


# =========================
# Leg-UP 核心流程
# =========================
def _legup_generate(data: pd.DataFrame,
                    target_item_list: List[int],
                    num_items: int,
                    attack_num: int,
                    cfg: LegUPConfig) -> np.ndarray:
    _set_seed(cfg.seed)
    device = cfg.device

    num_users = int(data['user'].max()) + 1
    X_real = _dense_matrix(num_users, num_items, data, device)  # [U,I]
    user_items, user_ratings, item_mean, item_pop = _build_indices(data, num_users)

    # 代理模型预热（用真实数据少量迭代得到 U0,V0）
    u_idx = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i_idx = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r_val = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)
    U0 = torch.randn(num_users, cfg.latent_dim, device=device) * 0.01
    V0 = torch.randn(num_items, cfg.latent_dim, device=device) * 0.01
    U0.requires_grad_(True); V0.requires_grad_(True)
    for _ in range(5):
        pred = (U0[u_idx] * V0[i_idx]).sum(-1)
        loss = F.mse_loss(pred, r_val) + cfg.reg * (U0.pow(2).mean() + V0.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U0, V0], create_graph=False)
        with torch.no_grad():
            U0 -= cfg.inner_lr * gU; V0 -= cfg.inner_lr * gV
    U0 = U0.detach(); V0 = V0.detach()

    # 选 selected items（热门且避开 target）
    counts = data['item'].value_counts()
    popular = [int(x) for x in counts.index.tolist() if int(x) not in set(target_item_list)]
    if len(popular) < cfg.num_selected:
        popular = [i for i in range(num_items) if i not in set(target_item_list)] + popular
    selected = popular[:cfg.num_selected]
    S = torch.tensor(selected, dtype=torch.long, device=device)
    T = torch.tensor(target_item_list, dtype=torch.long, device=device)

    # 模板用户：至少 P 个交互
    P = int(cfg.profile_size)
    candidates = _users_with_at_least(user_items, P)
    if not candidates:
        raise RuntimeError("No users with enough ratings to serve as templates.")

    # 模型
    G = Generator(num_items, len(selected), cfg).to(device)
    D = Discriminator(num_items).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d)
    bce = nn.BCEWithLogitsLoss()

    # —— 训练 —— #
    steps_per_epoch = max(1, attack_num // max(1, cfg.batch_size))
    for _ in range(cfg.epochs):
        for __ in range(steps_per_epoch):
            # ===== 判别器 =====
            for _k in range(cfg.k_disc):
                B = min(cfg.batch_size, len(candidates))
                users = np.random.choice(candidates, size=B, replace=True)
                x_real = X_real[torch.tensor(users, dtype=torch.long, device=device)]  # [B,I]

                # 组装假样本（模板 + 生成器）
                x_in = torch.zeros_like(x_real)
                filler_need = max(0, P - len(selected) - len(target_item_list))
                for bi, u in enumerate(users):
                    fillers = _sample_fillers(int(u), cfg, user_items, user_ratings, item_mean, item_pop, need=filler_need)
                    x_in[bi, fillers] = X_real[u, fillers]
                # 生成 selected 的连续离散值（0..5），再映射到 {1..5}
                g_disc, g_pref, tau_cum = G(x_in)            # [B,|S|]
                # 评分：Hsum∈[0..5] → {1..5} 的期望（+1），保持连续便于反传
                g_scores = (g_disc + 1.0).clamp(1.0, 5.0)
                x_fake = x_in.clone()
                x_fake.scatter_(1, S.unsqueeze(0).expand_as(g_scores), g_scores)
                # target 设为满分（push attack）
                x_fake.index_put_((torch.arange(B, device=device).unsqueeze(1),
                                   T.unsqueeze(0).expand(B, -1)),
                                  torch.full((B, len(T)), cfg.rating_max, device=device))

                # 判别器优化
                logits_real = D(x_real)
                logits_fake = D(x_fake.detach())
                loss_d = bce(logits_real, torch.ones_like(logits_real)) + \
                         bce(logits_fake, torch.zeros_like(logits_fake))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ===== 生成器 =====
            for _k in range(cfg.k_gen):
                B = min(cfg.batch_size, len(candidates))
                users = np.random.choice(candidates, size=B, replace=True)
                x_in = torch.zeros((B, num_items), dtype=torch.float32, device=device)
                filler_need = max(0, P - len(selected) - len(target_item_list))
                # 记录 selected 的“可重构标签”
                recon_target = torch.zeros((B, len(selected)), dtype=torch.float32, device=device)
                recon_mask   = torch.zeros_like(recon_target, dtype=torch.bool)

                for bi, u in enumerate(users):
                    fillers = _sample_fillers(int(u), cfg, user_items, user_ratings, item_mean, item_pop, need=filler_need)
                    x_in[bi, fillers] = X_real[u, fillers]
                    # 若模板用户在 selected 上有真实评分，则用于重构
                    sel_vals = X_real[u, S]
                    mask = sel_vals > 0
                    if mask.any():
                        recon_mask[bi] = mask
                        recon_target[bi, mask] = sel_vals[mask]

                g_disc, g_pref, tau_cum = G(x_in)
                g_scores = (g_disc + 1.0).clamp(1.0, 5.0)

                x_fake = x_in.clone()
                x_fake.scatter_(1, S.unsqueeze(0).expand_as(g_scores), g_scores)
                x_fake.index_put_((torch.arange(B, device=device).unsqueeze(1),
                                   T.unsqueeze(0).expand(B, -1)),
                                  torch.full((B, len(T)), cfg.rating_max, device=device))

                # —— 对抗损失（骗过判别器）——
                logits_fake = D(x_fake)
                loss_adv = bce(logits_fake, torch.ones_like(logits_fake))

                # —— 直接生成损失：基于代理 MF 的推高目标项（近似两层优化）——
                U, V = _inner_mf_few_steps(U0, V0, u_idx, i_idx, r_val, x_fake, cfg)
                # 在一批真实用户上评估 softmax 交叉熵（Eq.8）
                real_eval = torch.tensor(np.random.choice(num_users, size=min(512, num_users), replace=False),
                                         dtype=torch.long, device=device)
                scores = (U[real_eval] @ V.T)  # [Beval, I]
                logit_t = scores[:, T]         # [Beval, |T|]
                loss_direct = -torch.log_softmax(scores, dim=1).gather(1, T.unsqueeze(0).expand(scores.size(0), -1)).mean()

                # —— 可选重构损失（增强拟态性）——
                if recon_mask.any():
                    loss_recon = F.mse_loss(g_scores[recon_mask], recon_target[recon_mask])
                else:
                    loss_recon = torch.tensor(0.0, device=device)

                loss_g = cfg.lambda_adv*loss_adv + cfg.lambda_direct*loss_direct + cfg.lambda_recon*loss_recon
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    # —— 生成 attack_num 个对抗样本并离散化落盘 —— #
    fake_profiles = np.zeros((attack_num, num_items), dtype=np.int16)
    gen_batch = min(cfg.batch_size, attack_num)
    ptr = 0
    while ptr < attack_num:
        B = min(gen_batch, attack_num - ptr)
        users = np.random.choice(candidates, size=B, replace=True)
        x_in = torch.zeros((B, num_items), dtype=torch.float32, device=device)
        filler_need = max(0, P - len(selected) - len(target_item_list))
        for bi, u in enumerate(users):
            fillers = _sample_fillers(int(u), cfg, user_items, user_ratings, item_mean, item_pop, need=filler_need)
            x_in[bi, fillers] = X_real[u, fillers]
        g_disc, _, _ = G(x_in)
        g_scores = (g_disc + 1.0).clamp(1.0, 5.0)
        x_fake = x_in.clone()
        x_fake.scatter_(1, S.unsqueeze(0).expand_as(g_scores), g_scores)
        x_fake.index_put_((torch.arange(B, device=device).unsqueeze(1),
                           T.unsqueeze(0).expand(B, -1)),
                          torch.full((B, len(T)), cfg.rating_max, device=device))
        # 仅对已评分位置离散化到 {1..5}，未评分保持 0
        rated = x_fake > 0
        x_round = torch.where(rated, x_fake.clamp(cfg.rating_min, cfg.rating_max).round(),
                              torch.zeros_like(x_fake))
        fake_profiles[ptr:ptr+B] = x_round.to(torch.int16).cpu().numpy()
        ptr += B
    return fake_profiles


class AttackSimulator:
    """
    一次攻击样本生成（Leg-UP）
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

        self.cfg = LegUPConfig(device=str(self.device))

    def run(self, save: bool = True, seed=None) -> Tuple[List[int], np.ndarray]:
        data = pd.read_csv(self.remained_path, sep=' ', header=None, names=['user','item','rating'])

        # 假用户数量：ceil(attack_size * #users)
        num_users = int(data['user'].max()) + 1
        attack_num = max(1, math.ceil(self.attack_size * num_users))

        # 目标物品
        target_item_list, _, _, _ = obtain_target_items(self.n_items, data, self.dataset_name, self.num_target_items, seed=seed)

        # 默认 profile_size 若未设置，可用数据集均值（如 ML-100K≈90）
        if self.cfg.profile_size <= 0:
            avg = int(data.groupby('user').size().mean())
            self.cfg.profile_size = max(avg, len(target_item_list) + self.cfg.num_selected)

        fake_profiles = _legup_generate(
            data=data,
            target_item_list=target_item_list,
            num_items=self.n_items,
            attack_num=attack_num,
            cfg=self.cfg
        )

        if save:
            os.makedirs(self.dataset_dir, exist_ok=True)
            npy_path = f"./results/{self.dataset_name}/legup_fake_profiles.npy"
            np.save(npy_path, fake_profiles)

        print(f'target_item_list: {list(map(int, target_item_list))}')

        return list(map(int, target_item_list)), fake_profiles

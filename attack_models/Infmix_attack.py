import os, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from attack_models.select_target_items import obtain_target_items

# =========================
# Utils
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

def _build_hist_mode(data: pd.DataFrame, num_items: int):
    # 每个物品的评分直方图，得到众数（mode），无评分则为0
    hist = np.zeros((num_items, 6), dtype=np.int64)  # index 0..5，其中0统计不到
    for i, r in data[['item','rating']].itertuples(index=False):
        hist[int(i), int(r)] += 1
    mode = np.zeros(num_items, dtype=np.int64)
    for j in range(num_items):
        if hist[j,1:].sum() == 0: mode[j] = 0
        else: mode[j] = np.argmax(hist[j,1:]) + 1
    return mode

def _avg_user_len(data: pd.DataFrame):
    return int(round(data.groupby('user').size().mean()))

def _flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])

def _unflatten_like(vec: torch.Tensor, like_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    outs = []
    offset = 0
    for t in like_tensors:
        n = t.numel()
        outs.append(vec[offset:offset+n].view_as(t))
        offset += n
    return outs


# =========================
# Local MF simulator + pretrain
# =========================
def _pretrain_mf(num_users, num_items, u_idx, i_idx, r_val, cfg: InfmixCfg):
    device = cfg.device
    U = torch.randn(num_users, cfg.latent_dim, device=device) * 0.01
    V = torch.randn(num_items, cfg.latent_dim, device=device) * 0.01
    U.requires_grad_(True); V.requires_grad_(True)
    for _ in range(cfg.pretrain_steps):
        pred = (U[u_idx] * V[i_idx]).sum(-1)
        loss = F.mse_loss(pred, r_val) + cfg.reg*(U.pow(2).mean()+V.pow(2).mean())
        gU, gV = torch.autograd.grad(loss, [U, V], create_graph=False, allow_unused=True)
        if gU is None: gU = torch.zeros_like(U)
        if gV is None: gV = torch.zeros_like(V)
        with torch.no_grad():
            U -= cfg.inner_lr * gU
            V -= cfg.inner_lr * gV
    U.requires_grad_(True); V.requires_grad_(True)
    return U, V


# =========================
# Attack loss (promotion)：-log softmax prob(target)
# =========================
def _attack_loss(U, V, target_idx: torch.Tensor, cfg: InfmixCfg):
    num_users = U.size(0)
    m = min(cfg.atk_eval_users, num_users)
    idx = np.random.choice(num_users, size=m, replace=False)
    sample_users = torch.tensor(idx, dtype=torch.long, device=U.device)
    S = U[sample_users] @ V.T                      # [m, I]
    logp = torch.log_softmax(S, dim=1)             # softmax over items
    return -logp[:, target_idx].mean()             # 平均多个 target 的 log prob


# =========================
# Train loss on a minibatch (for Hessian & HVP)
# =========================
def _train_loss_batch(U, V, u_idx_all, i_idx_all, r_val_all, cfg: InfmixCfg):
    n = u_idx_all.numel()
    b = min(cfg.hvp_batch, n)
    sel = torch.randint(0, n, (b,), device=U.device)
    u = u_idx_all[sel]; it = i_idx_all[sel]; r = r_val_all[sel]
    pred = (U[u] * V[it]).sum(-1)
    return F.mse_loss(pred, r) + cfg.reg*(U.pow(2).mean()+V.pow(2).mean())


# =========================
# Conjugate Gradient to solve (H+damp*I)s = g using HVP
# =========================
def _hvp(U, V, u_idx_all, i_idx_all, r_val_all, vec, cfg: InfmixCfg):
    # compute Hessian-vector product of training loss w.r.t [U,V]
    loss = _train_loss_batch(U, V, u_idx_all, i_idx_all, r_val_all, cfg)
    grads = torch.autograd.grad(loss, [U, V], create_graph=True)
    flat_grads = _flatten_tensors(list(grads))
    prod = (flat_grads * vec).sum()
    hv = torch.autograd.grad(prod, [U, V], retain_graph=False)
    flat_hv = _flatten_tensors(list(hv))
    return flat_hv + cfg.cg_damping * vec

def _conjugate_gradient(U, V, u_idx_all, i_idx_all, r_val_all, b, cfg: InfmixCfg):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = (r*r).sum()
    for _ in range(cfg.cg_iters):
        Ap = _hvp(U, V, u_idx_all, i_idx_all, r_val_all, p, cfg)
        alpha = rs_old / (p*Ap).sum().clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r*r).sum()
        if rs_new.sqrt() < 1e-6:
            break
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new
    return x


# =========================
# Compute I_atk at x_min via IHVP trick:
#   s = H^{-1} ∇_θ L_atk
#   I_atk = -(1/n) * ∇_x [ s^T ∇_θ L_train(x_min) ]   (size = #items)
# =========================
def _compute_Iatk(U, V, u_idx_all, i_idx_all, r_val_all,
                  x_min_user: int, x_min_vec: torch.Tensor,
                  target_idx: torch.Tensor, cfg: InfmixCfg) -> torch.Tensor:
    device = U.device
    n_obs = u_idx_all.numel()

    # 1) g = ∇_θ L_atk
    L_atk = _attack_loss(U, V, target_idx, cfg)
    gU, gV = torch.autograd.grad(L_atk, [U, V], create_graph=False)
    g = _flatten_tensors([gU, gV]).detach()

    # 2) s = H^{-1} g  (via CG with HVP)
    s = _conjugate_gradient(U, V, u_idx_all, i_idx_all, r_val_all, g, cfg).detach()

    # 3) dot = s^T ∇_θ L_train(x_min)    (x_min as differentiable vector)
    #    then I_atk = -(1/n) * ∇_x dot
    # build loss of x_min using only that user u0 = x_min_user
    # x_min_vec: [I] with values in {0..5}, requires_grad=True
    x = x_min_vec.clone().detach().to(device).requires_grad_(True)
    u0 = x_min_user

    # indices where x has ratings (>0)
    idx = torch.nonzero(x > 0, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        # 极端兜底：若为空则返回全零
        return torch.zeros_like(x)

    pred = (U[u0].unsqueeze(0) * V[idx]).sum(-1)
    r = x[idx]
    L_xmin = F.mse_loss(pred, r, reduction='mean')  # 无需再加reg到θ上

    gU_x, gV_x = torch.autograd.grad(L_xmin, [U, V], create_graph=True, allow_unused=True)
    if gU_x is None: gU_x = torch.zeros_like(U)
    if gV_x is None: gV_x = torch.zeros_like(V)
    flat_gx = _flatten_tensors([gU_x, gV_x])
    dot = (flat_gx * s).sum()

    # grad w.r.t x (only positions in idx contribute). Need to connect x into graph:
    # 为了把 x 正确连入图，把 L_xmin 写成函数形式已经实现；此处直接对 dot 求 ∂/∂x
    grad_x, = torch.autograd.grad(dot, x, retain_graph=False)
    Iatk = -(1.0 / max(1, int(n_obs))) * grad_x
    # 对未评分位置的估计值意义不大，后续仅通过 x' 的非零位置做点积
    return Iatk.detach()


# =========================
# Usermix 生成候选
# x' = Π(λ x_i + (1-λ) x_j)，并限制非零数量为 m'
# =========================
def _usermix_candidates(X: torch.Tensor,  # [U, I] in {0..5}
                        base_user: int,
                        ncand: int,
                        mprime: int,
                        target_idx: List[int],
                        cfg: InfmixCfg) -> torch.Tensor:
    device = X.device
    U, I = X.shape
    xi = X[base_user]                       # [I]
    cands = torch.zeros((ncand, I), dtype=torch.float32, device=device)
    for k in range(ncand):
        j = random.randrange(U)
        xj = X[j]
        lam = np.random.beta(cfg.alpha, cfg.beta)
        xmix = lam*xi + (1-lam)*xj          # 连续
        # project: round to {0..5}，未评分保持 0
        xdisc = xmix.round().clamp_(0, cfg.rating_max)
        # 限制非零个数 = m'，但必须包含 target
        nonzero = torch.nonzero(xdisc > 0, as_tuple=False).squeeze(-1).tolist()
        # 目标物品确保在候选中
        for t in target_idx:
            if xdisc[t] <= 0: xdisc[t] = cfg.rating_max
        nonzero = torch.nonzero(xdisc > 0, as_tuple=False).squeeze(-1).tolist()
        if len(nonzero) > mprime:
            # 保留 target，随机保留其余
            keep = set(target_idx)
            rest = [p for p in nonzero if p not in keep]
            need = mprime - len(keep)
            if need > 0:
                pick = np.random.choice(rest, size=need, replace=False).tolist()
                keep.update(pick)
            mask = torch.zeros(I, dtype=torch.bool, device=device)
            mask[list(keep)] = True
            xdisc = torch.where(mask, xdisc, torch.zeros_like(xdisc))
        cands[k] = xdisc
    return cands  # [ncand, I] in {0..5}


# =========================
# Infmix main (attack only)
# =========================
def _infmix_generate(data: pd.DataFrame,
                     target_item_list: List[int],
                     num_items: int,
                     attack_num: int,
                     cfg: InfmixCfg) -> np.ndarray:
    _set_seed(cfg.seed)
    device = cfg.device
    # ----- data -----
    num_users = int(data['user'].max()) + 1
    X = _dense(num_users, num_items, data, device)     # [U,I] 真实评分 {0..5}
    u_idx_all = torch.tensor(data['user'].values, dtype=torch.long, device=device)
    i_idx_all = torch.tensor(data['item'].values, dtype=torch.long, device=device)
    r_val_all = torch.tensor(data['rating'].values, dtype=torch.float32, device=device)

    # ----- local MF -----
    U, V = _pretrain_mf(num_users, num_items, u_idx_all, i_idx_all, r_val_all, cfg)

    # ----- x_min 选择（近似）：选“评分最多”的真实用户，并把 target 打满分 -----
    user_len = (X > 0).sum(1)
    x_min_user = int(torch.argmax(user_len).item())
    x_min_vec = X[x_min_user].clone()
    for t in target_item_list:
        x_min_vec[t] = cfg.rating_max

    # ----- 影响向量 I_atk 计算（一次离线） -----
    target_tensor = torch.tensor(target_item_list, dtype=torch.long, device=device)
    Iatk = _compute_Iatk(U, V, u_idx_all, i_idx_all, r_val_all,
                         x_min_user, x_min_vec, target_tensor, cfg)  # [I]
    Iatk = Iatk.detach()

    # ----- 迭代生成 attack_num 个假用户 -----
    avg_len = _avg_user_len(data)
    mprime = max(len(target_item_list), int(round(avg_len * cfg.profile_size_ratio)))

    fake_profiles = np.zeros((attack_num, num_items), dtype=np.int16)
    for a in range(attack_num):
        # 固定一个基用户，生成 ncand 个候选，打分= Iatk·x'
        base = random.randrange(num_users)
        cands = _usermix_candidates(X, base, cfg.ncand, mprime, target_item_list, cfg)  # [ncand, I]
        # ensure targets = max
        if len(target_item_list) > 0:
            cands[:, target_tensor] = cfg.rating_max
        # 评分：Iatk dot x'（只对非零项有效）
        scores = (cands * Iatk.unsqueeze(0)).sum(1)
        best = torch.argmax(scores).item()
        x_best = cands[best].round().clamp_(0, cfg.rating_max).to(torch.int16).cpu().numpy()
        fake_profiles[a] = x_best

    return fake_profiles


class AttackSimulator:
    """
    Infmix 攻击（仅攻击部分）
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

        self.cfg = InfmixCfg(device=str(self.device))

    def run(self, save: bool = True, seed=None) -> Tuple[List[int], np.ndarray]:
        data = pd.read_csv(self.remained_path, sep=' ', header=None, names=['user','item','rating'])

        num_users = int(data['user'].max()) + 1
        attack_num = max(1, math.ceil(self.attack_size * num_users))

        # 选择 target items
        target_item_list, _, _, _ = obtain_target_items(self.n_items, data, self.dataset_name, self.num_target_items, seed=seed)

        fake_profiles = _infmix_generate(
            data=data,
            target_item_list=target_item_list,
            num_items=self.n_items,
            attack_num=attack_num,
            cfg=self.cfg
        )

        if save:
            os.makedirs(self.dataset_dir, exist_ok=True)
            npy_path = f"./results/{self.dataset_name}/infmix_fake_profiles.npy"
            np.save(npy_path, fake_profiles)

        print(f'target_item_list: {list(map(int, target_item_list))}')

        return list(map(int, target_item_list)), fake_profiles

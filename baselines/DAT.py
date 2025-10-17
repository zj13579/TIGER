import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import torch


# =========================
# DAT 主类（封装全部计算）
# =========================
class DAT_TIA:
    def __init__(
        self,
        dataset_name: str,
        cfg: DATConfig,
        combined_dataset_filename: str = "combined_dataset.txt"
    ):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.combined_dataset_filename = combined_dataset_filename

        # 最近一次 run 的中间结果（便于外部调试/取用）
        self.last_item_scores: Optional[pd.DataFrame] = None
        self.last_targets: Optional[List[int]] = None

        # 运行期内部变量
        self.df: Optional[pd.DataFrame] = None
        self.user_id_map: Optional[dict] = None
        self.item_id_map: Optional[dict] = None
        self.item_id_invmap: Optional[dict] = None
        self.num_users: int = 0
        self.num_items: int = 0
        self.R = None  # 用户-物品评分稀疏矩阵 CSR

    # ---------- 随机种子 ----------
    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # ---------- 数据加载 ----------
    def _load_combined(self, combined_path: str) -> None:
        df = pd.read_csv(
            combined_path,
            sep=' ',
            header=None,
            names=['user', 'item', 'rating']
        )
        u_ids = df['user'].unique()
        i_ids = df['item'].unique()
        self.user_id_map = {u: idx for idx, u in enumerate(sorted(u_ids))}
        self.item_id_map = {i: idx for idx, i in enumerate(sorted(i_ids))}
        self.item_id_invmap = {v: k for k, v in self.item_id_map.items()}

        df['u_idx'] = df['user'].map(self.user_id_map)
        df['i_idx'] = df['item'].map(self.item_id_map)

        self.num_users = len(self.user_id_map)
        self.num_items = len(self.item_id_map)
        self.df = df

        # 用户-物品评分稀疏矩阵
        self.R = sparse.coo_matrix(
            (df['rating'].values, (df['u_idx'].values, df['i_idx'].values)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        ).tocsr()

    # ---------- 构图：物品-物品共同用户 + 评分余弦 ----------
    def _build_item_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        assert self.R is not None

        # 共同用户计数 C = (R>0)^T (R>0)
        U_bin = self.R.copy()
        U_bin.data[:] = 1.0
        C = (U_bin.T @ U_bin).astype(np.float32).tocsr()
        deg = np.asarray(C.diagonal()).astype(np.float32)  # 每个物品被评分次数

        # 评分向量余弦相似度（I≈1682 → 稠密可行）
        S = cosine_similarity(self.R.T)  # ndarray [I, I]
        S = np.nan_to_num(S, nan=0.0)
        S_pos = np.maximum(S, 0.0).astype(np.float32)

        # 共同用户占比矩阵（按行 i 归一）
        deg_safe = np.maximum(deg, 1.0)
        common_ratio = C.multiply(sparse.diags(1.0 / deg_safe)).tocsr().toarray().astype(np.float32)

        # 直接影响强度 Δ = φ * common_ratio + (1-φ) * S_pos
        Delta = cfg.phi * common_ratio + (1.0 - cfg.phi) * S_pos
        np.fill_diagonal(Delta, 0.0)

        # 行归一为转移矩阵 T
        row_sum = Delta.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        T = Delta / row_sum
        return T, deg

    # ---------- 初始证据：满分比例（含“无满分”惩罚 ψ） ----------
    def _extreme_seed(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        assert self.df is not None
        max_mask = (self.df['rating'].values >= cfg.rating_max - 1e-8).astype(np.float32)
        item_idx = self.df['i_idx'].values
        cnt_total = np.bincount(item_idx, minlength=self.num_items).astype(np.float32)
        cnt_max = np.bincount(item_idx, weights=max_mask, minlength=self.num_items).astype(np.float32)

        frac_max = np.zeros(self.num_items, dtype=np.float32)
        nonzero = cnt_total > 0
        frac_max[nonzero] = cnt_max[nonzero] / cnt_total[nonzero]
        has_max = cnt_max > 0
        frac_max_adj = frac_max.copy()
        frac_max_adj[~has_max] *= cfg.penalize_no_max
        return frac_max_adj, cnt_total

    # ---------- 保守传播 ----------
    def _conservative_propagation(self, T: np.ndarray, seed: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        f_prev = seed.astype(np.float32)
        for _ in range(cfg.max_iter):
            f_next = (1.0 - cfg.beta) * seed + cfg.beta * (f_prev @ T)
            if np.max(np.abs(f_next - f_prev)) < cfg.converge_eps:
                break
            f_prev = f_next
        return f_prev

    # ---------- 邻域差分（兼容稠密/稀疏） ----------
    @staticmethod
    def _neighborhood_deviation(f_inf: np.ndarray, T) -> np.ndarray:
        if sparse.issparse(T):
            neigh_avg = T.dot(f_inf)
            try:
                neigh_avg = neigh_avg.A1
            except AttributeError:
                neigh_avg = np.asarray(neigh_avg).ravel()
        else:
            neigh_avg = T @ f_inf
        d = np.maximum(f_inf - neigh_avg, 0.0)
        rng = d.max() - d.min()
        if rng > 0:
            d = (d - d.min()) / rng
        return d

    # ---------- 前景/风险态度的简化实现（均值参考 + tanh 增益） ----------
    def _prospect_like(self) -> np.ndarray:
        assert self.df is not None
        item_idx = self.df['i_idx'].values
        ratings = self.df['rating'].values.astype(np.float32)
        sum_r = np.bincount(item_idx, weights=ratings, minlength=self.num_items).astype(np.float32)
        cnt = np.bincount(item_idx, minlength=self.num_items).astype(np.float32)
        mu = np.zeros(self.num_items, dtype=np.float32)
        mask = cnt > 0
        mu[mask] = sum_r[mask] / cnt[mask]
        gain = np.tanh(ratings - mu[item_idx])
        gain_sum = np.bincount(item_idx, weights=gain, minlength=self.num_items).astype(np.float32)
        gain_avg = np.zeros(self.num_items, dtype=np.float32)
        gain_avg[mask] = gain_sum[mask] / cnt[mask]
        if gain_avg.max() > gain_avg.min():
            gain_avg = (gain_avg - gain_avg.min()) / (gain_avg.max() - gain_avg.min())
        gain_avg = np.maximum(gain_avg, 0.0)
        return gain_avg

    # ---------- 综合可疑度并排序 ----------
    def _rank_items(self) -> pd.DataFrame:
        T, deg = self._build_item_graph()
        frac_max_seed, cnt_total = self._extreme_seed()
        f_inf = self._conservative_propagation(T, frac_max_seed)
        d_influence = self._neighborhood_deviation(f_inf, T)
        prospect_score = self._prospect_like()

        # 冷门软惩罚
        cold_weight = np.ones_like(deg, dtype=np.float32)
        cold_weight[deg < self.cfg.min_item_degree] = 0.3

        # 双证据融合
        prospect = 0.6 * frac_max_seed + 0.4 * prospect_score
        if prospect.max() > prospect.min():
            prospect = (prospect - prospect.min()) / (prospect.max() - prospect.min())
        s = self.cfg.alpha * d_influence + (1.0 - self.cfg.alpha) * prospect
        s *= cold_weight

        item_ids = [self.item_id_invmap[i] for i in range(self.num_items)]
        stats = pd.DataFrame({
            'item': item_ids,
            'score_total': s,
            'score_influence': d_influence,
            'score_seed_fracMax': frac_max_seed,
            'score_prospect': prospect_score,
            'degree': deg,
            'n_ratings': cnt_total
        }).sort_values('score_total', ascending=False).reset_index(drop=True)

        self.last_item_scores = stats.copy()
        return stats

    # ---------- 计算 DR_kT ----------
    def compute_dr_kT(self, item_stats: pd.DataFrame, target_item_list: List[int], T: Optional[int] = None) -> Dict[str, float]:
        Ks = list(self.cfg.Ks)
        true_targets = set(target_item_list)
        T = len(true_targets) if T is None else T
        if T == 0:
            return {f"DR_{k}T": 0.0 for k in Ks}

        ranked_items = item_stats['item'].tolist()
        res = {}
        for k in Ks:
            topk = k * T
            cand = set(ranked_items[:topk])
            hit = len(cand & true_targets)
            res[f"DR_{k}T"] = round(hit / T, 4)
        return res

    # ---------- 单次全流程 ----------
    def run_once(
        self,
        seed: int,
        simulator_kwargs: dict,
        attack_type: str
    ) -> Dict[str, float]:
        self._set_seed(seed)

        # 动态导入项目依赖（与模板一致）
        from attack_models.Heuristic_attacks import AttackSimulator
        # from attack_models.PGA_attack import AttackSimulator
        # from attack_models.SGLD_attack import AttackSimulator
        # from attack_models.AUSH_attack import AttackSimulator
        # from attack_models.TrialAttack import AttackSimulator
        # from attack_models.Infmix_attack import AttackSimulator
        # from attack_models.LegUP_attack import AttackSimulator

        from utils.matrix_factorization import MatrixFactorization

        # (1) 生成攻击（保存目标 & 注入）
        simulator = AttackSimulator(**simulator_kwargs)
        target_item_list, _ = simulator.run(save=True)
        self.last_targets = list(target_item_list)

        # (2) 构造合成数据集（项目接口）
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        matrix_factorization = MatrixFactorization(
            n_items=simulator_kwargs.get('n_items', 1682),
            latent_factors=32,
            dataset_name=self.dataset_name,
            device=device
        )
        _ = matrix_factorization.construct_combined_dataset(attack_type=attack_type)

        # (3) 读取合成数据
        combined_path = f'./datasets/{self.dataset_name}/{self.combined_dataset_filename}'
        self._load_combined(combined_path)

        # (4) DAT 排序 → (5) DR_kT
        item_stats = self._rank_items()
        drs = self.compute_dr_kT(item_stats, target_item_list)

        # 统一四舍五入
        drs = {k: round(v, 4) for k, v in drs.items()}
        return drs

    # ---------- 多种子循环 + 汇总 ----------
    def run_multi(
        self,
        seeds: List[int],
        simulator_kwargs: dict,
        attack_type: str,
        return_detail: bool = False
    ) -> Dict[str, Dict[str, float]]:

        per_seed = []
        for s in seeds:
            drs = self.run_once(seed=s, simulator_kwargs=simulator_kwargs, attack_type=attack_type)
            per_seed.append({'seed': s, **drs})

        df = pd.DataFrame(per_seed).set_index('seed')
        mean_dict = {k: round(float(df[k].mean()), 4) for k in df.columns}
        std_dict = {k: round(float(df[k].std(ddof=0)), 4) for k in df.columns}

        out = {'mean': mean_dict, 'std': std_dict}
        if return_detail:
            out['per_seed'] = per_seed
        return out
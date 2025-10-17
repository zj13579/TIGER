import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# =========================
# RD-TIA 主类
# =========================
class RD_TIA:

    def __init__(
        self,
        dataset_name: str,
        cfg: RDTIAConfig,
        Ks: List[int] = [1, 2, 3, 4, 5],
        combined_dataset_filename: str = "combined_dataset.txt",
    ):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.Ks = Ks
        self.combined_dataset_filename = combined_dataset_filename

        self.last_user_scores: Optional[pd.DataFrame] = None
        self.last_item_stats: Optional[pd.DataFrame] = None
        self.last_targets: Optional[List[int]] = None

    # -------- 随机种子 --------
    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # -------- 基础函数 --------
    @staticmethod
    def _item_means(df: pd.DataFrame) -> pd.Series:
        return df.groupby('item')['rating'].mean()

    @staticmethod
    def _rdma(df: pd.DataFrame, item_means: pd.Series) -> pd.Series:
        x = df.merge(item_means.rename('m'), on='item', how='left')
        return (x['rating'] - x['m']).abs().groupby(x['user']).mean()

    @staticmethod
    def _build_mats(df: pd.DataFrame, item_means: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 M(残差)、R(是否评分)、users、items"""
        users = np.sort(df['user'].unique())
        items = np.sort(df['item'].unique())
        u2i = {u: i for i, u in enumerate(users)}
        it2j = {it: j for j, it in enumerate(items)}
        M = np.zeros((len(users), len(items)), dtype=np.float32)
        R = np.zeros((len(users), len(items)), dtype=np.int8)
        for u, it, r in df[['user', 'item', 'rating']].itertuples(index=False):
            ui, ij = u2i[u], it2j[it]
            M[ui, ij] = r - float(item_means.loc[it])
            R[ui, ij] = 1
        return M, R, users, items

    @staticmethod
    def _overlap_mask(R: np.ndarray, min_overlap: int) -> np.ndarray:
        return (R @ R.T) >= min_overlap

    @staticmethod
    def _z(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / sd if sd > 0 else pd.Series(0.0, index=s.index)

    # -------- DegSim / DegSim′ --------
    def _degsim_from_matrix(self, M: np.ndarray, R: np.ndarray, topk: int, min_overlap: int) -> np.ndarray:
        """行向量余弦 Top-k 平均相似度；用 R 做重叠过滤"""
        X = normalize(M, norm='l2', axis=1, copy=True)
        n = X.shape[0]
        k = min(topk + 1, n)
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute').fit(X)
        dists, idxs = nn.kneighbors(X, return_distance=True)
        ovmask = self._overlap_mask(R, min_overlap)

        out = np.zeros(n, dtype=np.float32)
        for u in range(n):
            jj = idxs[u]
            dd = dists[u]
            mask_not_self = (jj != u)
            jj = jj[mask_not_self]
            dd = dd[mask_not_self]

            mask_overlap = ovmask[u, jj]
            jj = jj[mask_overlap]
            dd = dd[mask_overlap]

            if jj.size:
                take = min(topk, jj.size)
                sim = 1.0 - dd[:take]
                out[u] = float(np.mean(sim))
            else:
                out[u] = 0.0
        return out

    def _rating_levels(self, df: pd.DataFrame) -> List[float]:
        return sorted(df['rating'].unique().tolist())

    def _degsim_prime(
        self,
        df: pd.DataFrame,
        users_sorted: np.ndarray,
        items_sorted: np.ndarray,
        topk: int,
        min_overlap: int
    ) -> np.ndarray:
        """DegSim′：逐等级构造二值矩阵并平均。"""
        u2i = {u: i for i, u in enumerate(users_sorted)}
        it2j = {it: j for j, it in enumerate(items_sorted)}
        levels = self._rating_levels(df)
        sims = []
        for lv in levels:
            B = np.zeros((len(users_sorted), len(items_sorted)), dtype=np.float32)
            sub = df[df['rating'] == lv][['user', 'item']]
            for u, it in sub.itertuples(index=False):
                B[u2i[u], it2j[it]] = 1.0
            R = (B > 0).astype(np.int8)
            sims.append(self._degsim_from_matrix(B, R, topk=topk, min_overlap=min_overlap))
        sims = np.stack(sims, axis=1) if sims else np.zeros((len(users_sorted), 1), dtype=np.float32)
        return sims.mean(axis=1)

    # -------- RD-TIA 主计算 --------
    def rdtia(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        cfg = self.cfg
        item_means = self._item_means(df)
        M_res, R_ind, users, items = self._build_mats(df, item_means)
        rdma_s = self._rdma(df, item_means).reindex(users, fill_value=0.0)

        if cfg.use_degsim_prime:
            degsim_s = pd.Series(
                self._degsim_prime(df, users, items, cfg.topk_neighbors, cfg.min_overlap),
                index=users, name='degsim_prime'
            )
        else:
            degsim_s = pd.Series(
                self._degsim_from_matrix(M_res, R_ind, cfg.topk_neighbors, cfg.min_overlap),
                index=users, name='degsim'
            )

        score = cfg.alpha * self._z(rdma_s) + (1 - cfg.alpha) * (1 - self._z(degsim_s))
        thr = score.quantile(cfg.sus_user_quantile)
        sus_users = set(score[score >= thr].index)

        user_scores = pd.DataFrame({
            'user': users,
            'rdma': rdma_s.values,
            'deg': degsim_s.values,
            'score': score.values
        }).sort_values('score', ascending=False)

        tol = cfg.extreme_tolerance
        def is_ext(v: float) -> bool:
            return (v <= cfg.rating_min + tol) or (v >= cfg.rating_max - tol)

        df_sus = df[df['user'].isin(sus_users)]
        g = df_sus.groupby('item')['rating']
        n_total = g.size().rename('n_sus_rated')
        n_ext = g.apply(lambda s: int(np.sum([is_ext(x) for x in s.values]))).rename('n_extreme')
        item_stats = pd.concat([n_total, n_ext], axis=1).fillna(0)
        item_stats['ext_ratio'] = item_stats['n_extreme'] / item_stats['n_sus_rated'].replace(0, np.nan)
        item_stats['ext_ratio'] = item_stats['ext_ratio'].fillna(0.0)
        item_stats = item_stats.sort_values('ext_ratio', ascending=False).reset_index()

        self.last_user_scores = user_scores.copy()
        self.last_item_stats = item_stats.copy()

        return {'user_scores': user_scores, 'item_stats': item_stats}

    # -------- DR_kT --------
    def compute_dr_kT(self, item_stats: pd.DataFrame, target_item_list: List[int], T: Optional[int] = None) -> Dict[str, float]:
        Ks = self.Ks
        true_targets = set(target_item_list)
        T = len(true_targets) if T is None else T
        if T == 0:
            return {f"DR_{k}T": 0.0 for k in Ks}

        ranked = item_stats['item'].tolist()
        res = {}
        for k in Ks:
            topk = k * T
            cand = set(ranked[:topk])
            hit = len(cand & true_targets)
            res[f"DR_{k}T"] = round(hit / T, 4)
        return res

    # -------- 单次全流程（与 DeR_TIA 同风格） --------
    def run_once(self, seed: int, simulator_kwargs: dict, attack_type: str) -> Dict[str, float]:
        self._set_seed(seed)

        from attack_models.Heuristic_attacks import AttackSimulator
        # from attack_models.PGA_attack import AttackSimulator
        # from attack_models.SGLD_attack import AttackSimulator
        # from attack_models.AUSH_attack import AttackSimulator
        # from attack_models.LegUP_attack import AttackSimulator
        # from attack_models.TrialAttack import AttackSimulator
        # from attack_models.Infmix_attack import AttackSimulator

        from utils.matrix_factorization import MatrixFactorization

        simulator = AttackSimulator(**simulator_kwargs)
        target_item_list, _ = simulator.run(save=True)
        self.last_targets = list(target_item_list)

        # MatrixFactorization 实例化
        try:
            import torch
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        except Exception:
            device = 'cpu'

        n_items = simulator_kwargs.get('n_items', None)
        mf_kwargs = dict(dataset_name=self.dataset_name, device=device)
        if n_items is not None:
            mf_kwargs['n_items'] = n_items

        if 'latent_factors' not in mf_kwargs:
            mf_kwargs['latent_factors'] = 32

        matrix_factorization = MatrixFactorization(**mf_kwargs)
        _ = matrix_factorization.construct_combined_dataset(attack_type=attack_type)

        combined_path = f'./datasets/{self.dataset_name}/{self.combined_dataset_filename}'
        df = pd.read_csv(combined_path, sep=' ', header=None, names=['user', 'item', 'rating'])

        out = self.rdtia(df)
        drs = self.compute_dr_kT(out['item_stats'], target_item_list)
        return {k: round(v, 4) for k, v in drs.items()}

    # -------- 多种子循环 + 汇总 --------
    def run_multi(self, seeds: List[int], simulator_kwargs: dict, attack_type: str, return_detail: bool = False) -> Dict[str, Dict[str, float]]:
        per_seed = []
        for s in seeds:
            drs = self.run_once(seed=s, simulator_kwargs=simulator_kwargs, attack_type=attack_type)
            per_seed.append({'seed': s, **drs})

        df = pd.DataFrame(per_seed).set_index('seed')
        mean_dict = {k: round(float(df[k].mean()), 4) for k in df.columns}
        std_dict  = {k: round(float(df[k].std(ddof=0)), 4) for k in df.columns}

        out = {'mean': mean_dict, 'std': std_dict}
        if return_detail:
            out['per_seed'] = per_seed
        return out
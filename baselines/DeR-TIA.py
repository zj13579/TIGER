import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import torch

# =========================
# DeR-TIA 主类（封装全部计算）
# =========================
class DeR_TIA:

    def __init__(
        self,
        dataset_name: str,
        cfg: DerTIAConfig,
        Ks: List[int] = [1, 2, 3, 4, 5],
        combined_dataset_filename: str = "combined_dataset.txt"
    ):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.Ks = Ks
        self.combined_dataset_filename = combined_dataset_filename

        # 最近一次 run 的中间结果（便于外部调试/取用）
        self.last_user_scores: Optional[pd.DataFrame] = None
        self.last_item_stats: Optional[pd.DataFrame] = None
        self.last_targets: Optional[List[int]] = None

    # ---------- 随机种子 ----------
    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # 如需更强确定性，手动开启（可能变慢/不兼容）：
            # torch.use_deterministic_algorithms(True)
            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except Exception:
            pass

    # ---------- 基础子流程 ----------
    @staticmethod
    def _item_means(df: pd.DataFrame) -> pd.Series:
        return df.groupby('item')['rating'].mean()

    @staticmethod
    def _rdma(df: pd.DataFrame, item_means: pd.Series) -> pd.Series:
        x = df.merge(item_means.rename('m'), on='item', how='left')
        return (x['rating'] - x['m']).abs().groupby(x['user']).mean()

    @staticmethod
    def _build_mats(df: pd.DataFrame, item_means: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回：
          M: 残差矩阵 [#users, #items]
          R: 评分指示矩阵（是否评分）[#users, #items]
          users: 有序用户 id 列表
        """
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
        return M, R, users

    @staticmethod
    def _overlap_mask(R: np.ndarray, min_overlap: int) -> np.ndarray:
        # 使用 R（是否评分）计算重叠数，更稳健（避免“恰好等于均值导致残差=0”的误判）
        ov = R @ R.T
        return ov >= min_overlap

    @staticmethod
    def _z(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / sd if sd > 0 else pd.Series(0.0, index=s.index)

    def _degsim(self, M: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Top-k 近邻相似度（余弦），带最小重叠过滤。"""
        cfg = self.cfg
        X = normalize(M, norm='l2', axis=1, copy=True)
        n = X.shape[0]
        k = min(cfg.topk_neighbors + 1, n)
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute').fit(X)
        dists, idxs = nn.kneighbors(X, return_distance=True)  # 距离=1-相似度

        ovmask = self._overlap_mask(R, cfg.min_overlap)
        out = np.zeros(n, dtype=np.float32)

        for u in range(n):
            jj = idxs[u]
            dd = dists[u]
            # 去掉自身
            mask_not_self = (jj != u)
            jj = jj[mask_not_self]
            dd = dd[mask_not_self]

            # 保留与 u 有足够重叠的邻居
            mask_overlap = ovmask[u, jj]
            jj = jj[mask_overlap]
            dd = dd[mask_overlap]

            if jj.size:
                take = min(cfg.topk_neighbors, jj.size)
                sim = 1.0 - dd[:take]
                out[u] = float(np.mean(sim))
            else:
                out[u] = 0.0
        return out

    # ---------- DeR-TIA 主计算 ----------
    def der_tia(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        阶段一（用户打分异常 + 邻域一致性）→ 可疑用户集合；
        阶段二（可疑用户在各 item 上的极端评分占比 ext_ratio）→ item 排序。
        返回：
          - user_scores: [user, rdma, degsim, score]（按 score 降序）
          - item_stats : [item, n_sus_rated, n_extreme, ext_ratio]（按 ext_ratio 降序）
        """
        cfg = self.cfg
        item_means = self._item_means(df)
        M, R, users = self._build_mats(df, item_means)

        rdma_s = self._rdma(df, item_means).reindex(users, fill_value=0.0)
        degsim_s = pd.Series(self._degsim(M, R), index=users)

        score = cfg.alpha * self._z(rdma_s) + (1 - cfg.alpha) * (1 - self._z(degsim_s))
        thr = score.quantile(cfg.sus_user_quantile)
        sus_users = set(score[score >= thr].index)

        # 阶段二：在可疑用户集合上统计极端评分比
        tol = cfg.extreme_tolerance

        def is_ext(v: float) -> bool:
            return (v <= cfg.rating_min + tol) or (v >= cfg.rating_max - tol)

        df_sus = df[df['user'].isin(sus_users)]
        g = df_sus.groupby('item')['rating']
        n_total = g.size().rename('n_sus_rated')
        n_ext = g.apply(lambda s: int(np.sum([is_ext(x) for x in s.values]))).rename('n_extreme')
        stats = pd.concat([n_total, n_ext], axis=1).fillna(0)
        stats['ext_ratio'] = stats['n_extreme'] / stats['n_sus_rated'].replace(0, np.nan)
        stats['ext_ratio'] = stats['ext_ratio'].fillna(0.0)
        stats = stats.sort_values('ext_ratio', ascending=False).reset_index()

        user_scores = pd.DataFrame({
            'user': users,
            'rdma': rdma_s.values,
            'degsim': degsim_s.values,
            'score': score.values
        }).sort_values('score', ascending=False)

        # 记录最近一次结果（可选）
        self.last_user_scores = user_scores.copy()
        self.last_item_stats = stats.copy()

        return {'user_scores': user_scores, 'item_stats': stats}

    # ---------- 计算 DR_kT ----------
    def compute_dr_kT(
        self,
        item_stats: pd.DataFrame,
        target_item_list: List[int],
        T: Optional[int] = None
    ) -> Dict[str, float]:
        """
        DR_kT：在按 ext_ratio 排序的前 (k*T) 个候选中，命中真实目标的比例（对 T 归一）。
        """
        Ks = self.Ks
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

        # 加载项目内依赖，避免类被 import 时就硬依赖
        from attack_models.Heuristic_attacks import AttackSimulator
        # from attack_models.PGA_attack import AttackSimulator
        # from attack_models.SGLD_attack import AttackSimulator
        # from attack_models.AUSH_attack import AttackSimulator
        # from attack_models.LegUP_attack import AttackSimulator
        # from attack_models.TrialAttack import AttackSimulator
        # from attack_models.Infmix_attack import AttackSimulator

        from utils.matrix_factorization import MatrixFactorization

        # (1) 生成攻击（保存目标 & 注入）
        simulator = AttackSimulator(**simulator_kwargs)
        target_item_list, _ = simulator.run(save=True)
        self.last_targets = list(target_item_list)

        # (2) 构造合成数据集（项目接口）
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        matrix_factorization = MatrixFactorization(n_items=1682, latent_factors=32, dataset_name='ml-100k', device=device)
        _, _, _ = matrix_factorization.construct_combined_dataset(attack_type=attack_type)

        # (3) 读取合成数据
        combined_path = f'./datasets/{self.dataset_name}/{self.combined_dataset_filename}'
        df = pd.read_csv(combined_path, sep=' ', header=None, names=['user', 'item', 'rating'])

        # (4) DeR-TIA + (5) DR_kT
        out = self.der_tia(df)
        drs = self.compute_dr_kT(out['item_stats'], target_item_list)

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
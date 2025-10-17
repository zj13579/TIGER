import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import NearestNeighbors  # 备用

# =========================
# SVM-TIA 主类
# =========================
class SVM_TIA:

    def __init__(
        self,
        dataset_name: str,
        cfg: SVMTIAConfig,
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
        except Exception:
            pass

    # ---------- 工具 ----------
    @staticmethod
    def _item_means(df: pd.DataFrame) -> pd.Series:
        return df.groupby('item')['rating'].mean()

    @staticmethod
    def _build_residual(df: pd.DataFrame, item_means: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = np.sort(df['user'].unique())
        items = np.sort(df['item'].unique())
        u2i = {u: i for i, u in enumerate(users)}
        it2j = {it: j for j, it in enumerate(items)}
        M = np.zeros((len(users), len(items)), dtype=np.float32)
        for u, it, r in df[['user', 'item', 'rating']].itertuples(index=False):
            M[u2i[u], it2j[it]] = r - float(item_means.loc[it])
        return M, users, items

    @staticmethod
    def _overlap_counts(M_bin: np.ndarray) -> np.ndarray:
        return M_bin @ M_bin.T  # 共同评分数

    def _degsim_from_matrix(self, M: np.ndarray, topk: int, min_overlap: int) -> np.ndarray:
        """
        任意用户×物品矩阵的 Top-k 余弦相似度均值；无有效邻居 → 0。
        基于矩阵内积（无需 NN 拟合），并做重叠过滤与数值稳健处理。
        """
        if M.shape[0] == 0:
            return np.array([], dtype=np.float32)
        X = normalize(M, norm='l2', axis=1, copy=True)
        S = X @ X.T  # 余弦相似度
        np.fill_diagonal(S, -np.inf)
        ov = self._overlap_counts((M != 0).astype(np.float32))
        S[ov < min_overlap] = -np.inf
        n = S.shape[0]
        out = np.zeros(n, dtype=np.float32)
        for u in range(n):
            row = S[u]
            mask = np.isfinite(row)
            if not mask.any():
                out[u] = 0.0
                continue
            sims = row[mask]
            k = min(topk, sims.size)
            if k == 0:
                out[u] = 0.0
                continue
            idx = np.argpartition(-sims, k - 1)[:k]
            val = float(np.mean(sims[idx]))
            if not np.isfinite(val):
                val = 0.0
            out[u] = val
        return out

    @staticmethod
    def _rating_levels(df: pd.DataFrame) -> List[float]:
        return sorted(df['rating'].dropna().unique().tolist())

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
        if len(levels) == 0:
            return np.zeros(len(users_sorted), dtype=np.float32)
        sims = []
        for lv in levels:
            B = np.zeros((len(users_sorted), len(items_sorted)), dtype=np.float32)
            for u, it in df.loc[df['rating'] == lv, ['user', 'item']].itertuples(index=False):
                B[u2i[u], it2j[it]] = 1.0
            sims.append(self._degsim_from_matrix(B, topk=topk, min_overlap=min_overlap))
        sim_mat = np.stack(sims, axis=1)
        sim_avg = np.mean(sim_mat, axis=1)
        sim_avg = np.nan_to_num(sim_avg, nan=0.0, posinf=0.0, neginf=0.0)
        return sim_avg

    @staticmethod
    def _rdma(df: pd.DataFrame, item_means: pd.Series, users_sorted: np.ndarray) -> np.ndarray:
        x = df.merge(item_means.rename('m'), on='item', how='left')
        rd = (x['rating'] - x['m']).abs()
        rdma = rd.groupby(x['user']).mean()
        rdma = rdma.reindex(users_sorted).fillna(0.0).values
        return rdma

    def _user_basic_feats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        用户特征：RDMA, DegSim*, rated_cnt, max_ratio, min_ratio, mean_rating, var_rating
        无 NaN/Inf。
        """
        cfg = self.cfg
        item_means = self._item_means(df)
        M_res, users, items = self._build_residual(df, item_means)

        if cfg.use_degsim_prime:
            deg = self._degsim_prime(df, users, items, cfg.topk_neighbors, cfg.min_overlap)
            deg_name = 'degsim_prime'
        else:
            deg = self._degsim_from_matrix(M_res, cfg.topk_neighbors, cfg.min_overlap)
            deg_name = 'degsim'

        rdma = self._rdma(df, item_means, users)

        grp_u = df.groupby('user', sort=False)
        rated_cnt = grp_u.size().reindex(users).fillna(0).astype(float).values
        mean_rating = grp_u['rating'].mean().reindex(users).fillna(0.0).values
        var_rating = grp_u['rating'].var(ddof=0).reindex(users).fillna(0.0).values

        tol = cfg.extreme_tolerance
        is_max = (df['rating'] >= (cfg.rating_max - tol)).astype(float)
        is_min = (df['rating'] <= (cfg.rating_min + tol)).astype(float)
        max_ratio = grp_u.apply(lambda s: is_max.loc[s.index].mean()).reindex(users).fillna(0.0).values
        min_ratio = grp_u.apply(lambda s: is_min.loc[s.index].mean()).reindex(users).fillna(0.0).values

        feats = np.column_stack([rdma, deg, rated_cnt, max_ratio, min_ratio, mean_rating, var_rating]).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        cols = ['rdma', deg_name, 'rated_cnt', 'max_ratio', 'min_ratio', 'mean_rating', 'var_rating']
        return pd.DataFrame({'user': users}), pd.DataFrame(feats, columns=cols)

    @staticmethod
    def _tia_on_suspicious_users(
        df: pd.DataFrame, sus_users: List[int], rating_min: float, rating_max: float, extreme_tolerance: float
    ) -> pd.DataFrame:
        if len(sus_users) == 0:
            return pd.DataFrame(columns=['item', 'n_sus_rated', 'n_extreme', 'ext_ratio'])
        sub = df[df['user'].isin(sus_users)].copy()
        tol = extreme_tolerance

        def is_ext(v: float) -> bool:
            return (v <= rating_min + tol) or (v >= rating_max - tol)

        g = sub.groupby('item')['rating']
        n_total = g.size().rename('n_sus_rated')
        n_ext = g.apply(lambda s: np.sum([is_ext(x) for x in s.values])).rename('n_extreme')
        stats = pd.concat([n_total, n_ext], axis=1).fillna(0)
        stats['ext_ratio'] = stats['n_extreme'] / stats['n_sus_rated'].replace(0, np.nan)
        stats['ext_ratio'] = stats['ext_ratio'].fillna(0.0)
        stats = stats.replace([np.inf, -np.inf], 0.0)
        return stats.sort_values('ext_ratio', ascending=False).reset_index()

    # ---------- SVM-TIA 主计算 ----------
    def svm_tia(
        self,
        df: pd.DataFrame,
        attack_user_list: Optional[List[int]] = None
    ) -> Dict[str, pd.DataFrame]:
        user_df, Xdf = self._user_basic_feats(df)
        Xdf = Xdf.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X = Xdf.values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        users = user_df['user'].values
        cfg = self.cfg

        if attack_user_list is not None and len(attack_user_list) > 0:
            # 监督：SVC
            atk_set = set(attack_user_list)
            y = np.array([1 if u in atk_set else 0 for u in users], dtype=np.int32)
            clf = SVC(kernel=cfg.svm_kernel, C=cfg.svm_C, gamma=cfg.svm_gamma,
                      class_weight='balanced', probability=True, random_state=42)
            clf.fit(Xs, y)
            proba = clf.predict_proba(Xs)[:, 1]
            pred = (proba >= 0.5).astype(int)
            sus_users = users[pred == 1].tolist()
            user_scores = pd.DataFrame({'user': users, 'prob': proba, 'pred_suspicious': pred})
        else:
            # 无监督：One-Class SVM
            oc = OneClassSVM(kernel=cfg.ocsvm_kernel, gamma=cfg.ocsvm_gamma, nu=cfg.ocsvm_nu)
            oc.fit(Xs)
            oc_pred = oc.predict(Xs)  # +1 正常 / -1 异常
            sus_users = users[oc_pred == -1].tolist()
            dfun = oc.decision_function(Xs).ravel()
            dfun_norm = (dfun.min() - dfun) / (dfun.min() - dfun.max() + 1e-8)
            user_scores = pd.DataFrame({'user': users, 'ocsvm_score': dfun_norm,
                                        'pred_suspicious': (oc_pred == -1).astype(int)})

        item_stats = self._tia_on_suspicious_users(
            df, sus_users,
            rating_min=cfg.rating_min, rating_max=cfg.rating_max,
            extreme_tolerance=cfg.extreme_tolerance
        )
        sort_col = 'prob' if 'prob' in user_scores.columns else 'ocsvm_score'
        user_scores = user_scores.sort_values(sort_col, ascending=False)

        self.last_user_scores = user_scores.copy()
        self.last_item_stats = item_stats.copy()

        return {'user_scores': user_scores, 'item_stats': item_stats}

    # ---------- DR_kT ----------
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

    # ---------- 单次全流程 ----------
    def run_once(
        self,
        seed: int,
        simulator_kwargs: dict,
        attack_type: str,
        attack_user_list: Optional[List[int]] = None
    ) -> Dict[str, float]:

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

        out = self.svm_tia(df, attack_user_list=attack_user_list)
        drs = self.compute_dr_kT(out['item_stats'], target_item_list)
        return {k: round(v, 4) for k, v in drs.items()}

    # ---------- 多种子循环 + 汇总 ----------
    def run_multi(
        self,
        seeds: List[int],
        simulator_kwargs: dict,
        attack_type: str,
        attack_user_list: Optional[List[int]] = None,
        return_detail: bool = False
    ) -> Dict[str, Dict[str, float]]:
        per_seed = []
        for s in seeds:
            print(f'----------seed={s}, start----------')
            drs = self.run_once(
                seed=s,
                simulator_kwargs=simulator_kwargs,
                attack_type=attack_type,
                attack_user_list=attack_user_list
            )
            per_seed.append({'seed': s, **drs})

        df = pd.DataFrame(per_seed).set_index('seed')
        mean_dict = {k: round(float(df[k].mean()), 4) for k in df.columns}
        std_dict  = {k: round(float(df[k].std(ddof=0)), 4) for k in df.columns}

        out = {'mean': mean_dict, 'std': std_dict}
        if return_detail:
            out['per_seed'] = per_seed
        return out
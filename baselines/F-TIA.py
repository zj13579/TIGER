import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# =========================
# F-TIA 主类
# =========================
class F_TIA:

    def __init__(
        self,
        dataset_name: str,
        cfg: FTIAConfig,
        Ks: List[int] = [1, 2, 3, 4, 5],
        combined_dataset_filename: str = "combined_dataset.txt",
    ):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.Ks = Ks
        self.combined_dataset_filename = combined_dataset_filename

        self.last_item_feats: Optional[pd.DataFrame] = None
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
    def _safe_series(x: pd.Series, fill: float = 0.0) -> pd.Series:
        return x.replace([np.inf, -np.inf], np.nan).fillna(fill)

    @staticmethod
    def _entropy_normalized(counts: np.ndarray) -> float:
        total = counts.sum()
        L = (counts > 0).sum()
        if total <= 0 or L <= 1:
            return 0.0
        p = counts[counts > 0] / total
        H = -np.sum(p * np.log(p + 1e-12))
        return float(H / np.log(L + 1e-12))

    # ---------- 构建物品特征 ----------
    def _build_item_features(self, df: pd.DataFrame) -> pd.DataFrame:
        assert {'user', 'item', 'rating'}.issubset(df.columns)
        users_total = df['user'].nunique()
        tol = self.cfg.extreme_tolerance

        grp = df.groupby('item', sort=False)
        cnt = grp.size().rename('n_rated')
        mean_rating = grp['rating'].mean().rename('mean_rating')
        var_rating = grp['rating'].var(ddof=0).rename('var_rating')
        std_rating = grp['rating'].std(ddof=0).rename('std_rating')

        # MAD
        tmp = df.merge(mean_rating.rename('mu_i'), left_on='item', right_index=True, how='left')
        tmp['abs_dev'] = (tmp['rating'] - tmp['mu_i']).abs()
        mad = tmp.groupby('item', sort=False)['abs_dev'].mean().rename('mad').reindex(mean_rating.index)

        # 极端分比例（无 FutureWarning 的写法）
        prop_max = (
            df.assign(is_max=(df['rating'] >= (self.cfg.rating_max - tol)).astype(int))
              .groupby('item', sort=False)['is_max']
              .mean()
              .rename('prop_max')
        )
        prop_min = (
            df.assign(is_min=(df['rating'] <= (self.cfg.rating_min + tol)).astype(int))
              .groupby('item', sort=False)['is_min']
              .mean()
              .rename('prop_min')
        )
        prop_extreme = (prop_max + prop_min).rename('prop_extreme')
        prop_imb = (prop_max - prop_min).abs().rename('prop_maxmin_imbalance')

        # 等级分布熵
        levels = sorted(df['rating'].dropna().unique().tolist())
        level_counts = {}
        for lv in levels:
            level_counts[lv] = (
                df.assign(ind=(df['rating'] == lv).astype(int))
                  .groupby('item', sort=False)['ind'].sum()
            )
        level_counts_df = pd.DataFrame(level_counts).reindex(mean_rating.index).fillna(0.0)
        entropy_norm = level_counts_df.apply(lambda row: self._entropy_normalized(row.values), axis=1).rename('entropy_norm')
        entropy_neg = (1.0 - entropy_norm).rename('entropy_neg')

        # 偏度/峰度
        skew = grp['rating'].apply(lambda s: float(pd.Series(s.values).skew())).rename('skew')
        kurt = grp['rating'].apply(lambda s: float(pd.Series(s.values).kurt())).rename('kurtosis')
        skew_abs = np.abs(skew).rename('skew_abs')

        # 覆盖率与 z 分
        coverage = (cnt / max(users_total, 1)).rename('coverage')

        def _z(sr: pd.Series) -> pd.Series:
            mu, sd = sr.mean(), sr.std(ddof=0)
            if sd <= 1e-12:
                return pd.Series(0.0, index=sr.index)
            return (sr - mu) / (sd + 1e-12)

        z_pop = _z(cnt).rename('z_popularity')
        z_mean = _z(mean_rating).rename('z_mean_rating')

        feats = pd.concat([
            cnt, coverage, mean_rating, var_rating, std_rating, mad,
            prop_max, prop_min, prop_extreme, prop_imb,
            entropy_norm, entropy_neg, skew, skew_abs, kurt,
            z_pop, z_mean
        ], axis=1)

        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feats.reset_index()  # ['item', ...features...]

    # ---------- 打分（加权 / IsolationForest） ----------
    def _weighted_score(self, item_feats: pd.DataFrame) -> pd.DataFrame:
        feats = item_feats.copy()

        def _zcol(col):
            v = feats[col].values
            mu, sd = np.mean(v), np.std(v)
            feats[col + '_z'] = 0.0 if sd <= 1e-12 else (v - mu) / (sd + 1e-12)

        for col in ['prop_extreme','prop_maxmin_imbalance','mad','entropy_neg','skew_abs','kurtosis','z_popularity','z_mean_rating']:
            if col not in feats.columns:
                feats[col] = 0.0
            _zcol(col)

        c = self.cfg
        feats['ftia_score'] = (
            c.w_prop_extreme          * feats['prop_extreme_z'] +
            c.w_prop_maxmin_imbalance * feats['prop_maxmin_imbalance_z'] +
            c.w_mad                   * feats['mad_z'] +
            c.w_entropy_neg           * feats['entropy_neg_z'] +
            c.w_skew_abs              * feats['skew_abs_z'] +
            c.w_kurtosis              * feats['kurtosis_z'] +
            c.w_popularity_z          * feats['z_popularity_z'] +
            c.w_mean_rating_z         * feats['z_mean_rating_z']
        )
        return feats

    def _isoforest_score(self, item_feats: pd.DataFrame) -> pd.DataFrame:
        feats = item_feats.copy()
        model_cols = [
            'n_rated','coverage','mean_rating','var_rating','std_rating','mad',
            'prop_max','prop_min','prop_extreme','prop_maxmin_imbalance',
            'entropy_norm','entropy_neg','skew','skew_abs','kurtosis',
            'z_popularity','z_mean_rating'
        ]
        for c in model_cols:
            if c not in feats.columns:
                feats[c] = 0.0
        X = feats[model_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        iso = IsolationForest(
            n_estimators=self.cfg.if_n_estimators,
            contamination=self.cfg.if_contamination,
            random_state=self.cfg.if_random_state
        )
        iso.fit(Xs)
        feats['ftia_score'] = -iso.score_samples(Xs)  # 越大越可疑
        return feats

    # ---------- F-TIA 主计算 ----------
    def ftia(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        item_feats = self._build_item_features(df)
        if self.cfg.model == "isoforest":
            scored = self._isoforest_score(item_feats)
        else:
            scored = self._weighted_score(item_feats)

        keep_cols = [c for c in scored.columns if c in
                     ['item','ftia_score','n_rated','prop_extreme','prop_max','prop_min','mad','entropy_neg','z_popularity','z_mean_rating']]
        item_stats = scored[keep_cols].sort_values('ftia_score', ascending=False).reset_index(drop=True)

        self.last_item_feats = scored.copy()
        self.last_item_stats = item_stats.copy()
        return {'item_feats': scored, 'item_stats': item_stats}

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

        # MatrixFactorization（device 兜底）
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

        mf = MatrixFactorization(**mf_kwargs)
        _ = mf.construct_combined_dataset(attack_type=attack_type)

        combined_path = f'./datasets/{self.dataset_name}/{self.combined_dataset_filename}'
        df = pd.read_csv(combined_path, sep=' ', header=None, names=['user', 'item', 'rating'])

        out = self.ftia(df)
        drs = self.compute_dr_kT(out['item_stats'], target_item_list)
        return {k: round(v, 4) for k, v in drs.items()}

    # ---------- 多种子循环 + 汇总 ----------
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

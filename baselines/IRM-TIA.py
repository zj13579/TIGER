import os
import random
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch

# =========================
# IRM-TIA 无时间戳版封装
# =========================
class IRM_TIA:

    def __init__(
        self,
        dataset_name: str,
        cfg: IRMTIAConfig,
        Ks: List[int] = [1, 2, 3, 4, 5]
    ):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.Ks = Ks
        # 最近一次 run 的中间结果（可选）
        self.last_user_features: Optional[pd.DataFrame] = None
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

    # ---------- 基础统计 ----------
    @staticmethod
    def _basic_stats(df: pd.DataFrame) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame]:
        r_global = df['rating'].mean()
        r_min = df['rating'].min()
        r_max = df['rating'].max()
        # item: pop/mean/var
        item_stats = df.groupby('item')['rating'].agg(['count', 'mean', 'var']).rename(
            columns={'count': 'pop', 'mean': 'mean', 'var': 'var'})
        item_stats['var'] = item_stats['var'].fillna(0.0)
        # user: activity/mean/var
        user_stats = df.groupby('user')['rating'].agg(['count', 'mean', 'var']).rename(
            columns={'count': 'activity', 'mean': 'mean', 'var': 'var'})
        user_stats['var'] = user_stats['var'].fillna(0.0)
        return r_global, r_min, r_max, item_stats, user_stats

    # ---------- Kulc 共现 + 评分分布相似度（JS-based） ----------
    @staticmethod
    def _item_users(df: pd.DataFrame) -> Dict[int, set]:
        return df.groupby('item')['user'].apply(lambda s: set(s.tolist())).to_dict()

    @staticmethod
    def _item_hist(df: pd.DataFrame, r_min: float, r_max: float) -> Dict[int, np.ndarray]:
        """离散评分直方图（假定整数评分；若非整数，可自行分桶）"""
        bins = int(round(r_max - r_min)) + 1
        out = {}
        for it, s in df.groupby('item')['rating']:
            vec = np.zeros(bins, dtype=float)
            for r in s.values:
                idx = int(round(r - r_min))
                idx = max(0, min(bins - 1, idx))
                vec[idx] += 1
            sm = vec.sum()
            out[it] = (vec / sm) if sm > 0 else np.ones(bins) / bins
        return out

    @staticmethod
    def _js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        p = p + eps; q = q + eps
        m = 0.5 * (p + q)
        def kld(a, b):
            return float(np.sum(a * np.log(a / b)))
        return 0.5 * (kld(p, m) + kld(q, m))

    def _pairwise_item_sims(
        self, df: pd.DataFrame, r_min: float, r_max: float
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """返回：kulc[(i,j)], js_sim[(i,j)]"""
        item_users = self._item_users(df)
        items = list(item_users.keys())
        pop = {i: len(us) for i, us in item_users.items()}
        hist = self._item_hist(df, r_min, r_max)

        kulc, js_sim = {}, {}
        eps = 1e-8
        for a_idx in range(len(items)):
            ia = items[a_idx]
            Ua = item_users[ia]; na = pop[ia]
            ha = hist[ia]
            for b_idx in range(a_idx + 1, len(items)):
                ib = items[b_idx]
                Ub = item_users[ib]; nb = pop[ib]
                inter = len(Ua & Ub)
                if inter > 0:
                    val = 0.5 * (inter / (na + eps) + inter / (nb + eps))
                    kulc[(ia, ib)] = val
                    kulc[(ib, ia)] = val
                # JS-based similarity
                hb = hist[ib]
                js = self._js_div(ha, hb)
                js_sim[(ia, ib)] = 1.0 / (1.0 + js)
                js_sim[(ib, ia)] = js_sim[(ia, ib)]
        return kulc, js_sim

    # ---------- 用户行为特征 ----------
    def _user_features(
        self, df: pd.DataFrame, r_global: float, item_stats: pd.DataFrame, user_stats: pd.DataFrame,
        kulc: Dict[Tuple[int, int], float], js_sim: Dict[Tuple[int, int], float]
    ) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
        """
        返回：
          feat_df：每用户一行（RDUT、SDUT、activity、uvar、umean、avg_pop、avg_item_var）
          X      ：特征矩阵（numpy）
          user_ids：有序用户列表（与 X 对齐）
        """
        users = np.sort(df['user'].unique())
        rows = []
        for u in users:
            sub = df[df['user'] == u]
            items = sub['item'].unique().tolist()
            m = len(items)
            # pairwise RDUT/SDUT
            k_vals, js_vals = [], []
            for i in range(m):
                ia = items[i]
                for j in range(i + 1, m):
                    ib = items[j]
                    if (ia, ib) in kulc: k_vals.append(kulc[(ia, ib)])
                    if (ia, ib) in js_sim: js_vals.append(js_sim[(ia, ib)])
            rdut = float(np.mean(k_vals)) if len(k_vals) else 0.0
            sdut = float(np.mean(js_vals)) if len(js_vals) else 0.0

            # 统计特征（补偿无时间窗）
            if self.cfg.use_stats_features:
                activity = float(user_stats.loc[u, 'activity']) if u in user_stats.index else 0.0
                uvar = float(user_stats.loc[u, 'var']) if u in user_stats.index else 0.0
                umean = float(user_stats.loc[u, 'mean']) if u in user_stats.index else r_global
                pop_list = [float(item_stats.loc[it, 'pop']) if it in item_stats.index else 0.0 for it in items]
                ivar_list = [float(item_stats.loc[it, 'var']) if it in item_stats.index else 0.0 for it in items]
                avg_pop = float(np.mean(pop_list)) if len(pop_list) else 0.0
                avg_item_var = float(np.mean(ivar_list)) if len(ivar_list) else 0.0
                rows.append([u, rdut, sdut, activity, uvar, umean, avg_pop, avg_item_var])
            else:
                rows.append([u, rdut, sdut])

        if self.cfg.use_stats_features:
            cols = ['user', 'RDUT', 'SDUT', 'activity', 'uvar', 'umean', 'avg_pop', 'avg_item_var']
        else:
            cols = ['user', 'RDUT', 'SDUT']
        feat_df = pd.DataFrame(rows, columns=cols)

        X = feat_df.drop(columns=['user']).values.astype(np.float32)
        return feat_df, X, list(users)

    # ---------- 可疑用户识别（PCA + PR/BVL + KMeans） ----------
    def _suspicious_users(self, X: np.ndarray, users: List[int]) -> List[int]:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # PCA: 保留方差 >= (1 - epsilon)
        pca_all = PCA(n_components=min(Xs.shape[0], Xs.shape[1]))
        pca_all.fit(Xs)
        cum = pca_all.explained_variance_ratio_.cumsum()
        keep_ratio = max(0.0, 1.0 - self.cfg.epsilon)
        k = int(np.searchsorted(cum, keep_ratio) + 1)
        k = max(1, min(k, Xs.shape[1]))
        pca = PCA(n_components=k)
        Z = pca.fit_transform(Xs)
        Xhat = pca.inverse_transform(Z)

        PR = np.linalg.norm(Xs - Xhat, axis=1)
        BVL = np.linalg.norm(Xs, axis=1)
        Z2 = np.vstack([PR, BVL]).T

        km = KMeans(n_clusters=2, random_state=42, n_init=self.cfg.kmeans_n_init)
        lab = km.fit_predict(Z2)
        bvl_mean = {c: float(BVL[lab == c].mean()) for c in np.unique(lab)}
        sus_cluster = max(bvl_mean.items(), key=lambda x: x[1])[0]
        sus_users = [users[i] for i in range(len(users)) if lab[i] == sus_cluster]

        # ---- 兜底：若嫌疑集合过小或为空，用 z(BVL)+z(PR) 的 Top-20% 替代 ----
        min_needed = max(3, int(0.01 * len(users)))  # 至少 1% 或 3 人
        if len(sus_users) < min_needed:
            B = pd.Series(BVL);
            P = pd.Series(PR)
            z = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8)
            combo = z(B) + z(P)
            thr = combo.quantile(0.80)
            sus_users = [users[i] for i in range(len(users)) if combo.iloc[i] >= thr]

        self.last_user_scores = pd.DataFrame({'user': users, 'PR': PR, 'BVL': BVL, 'cluster': lab})
        return sus_users

    # ---------- 目标物品识别（SBA × f_i） ----------
    @staticmethod
    def _compute_w(r_ui: float, r_u: float, r_i: float, r_g: float, eps: float = 1e-8) -> float:
        return 1.0 + (r_ui - r_u) / (r_u + eps) + (r_ui - r_i) / (r_i + eps) + (r_ui - r_g) / (r_g + eps)

    def _rank_items_by_P(self, df: pd.DataFrame, sus_users: List[int]) -> Tuple[pd.DataFrame, Dict[int, float]]:
        cfg = self.cfg
        if not sus_users:
            return pd.DataFrame(columns=['item', 'SBA', 'ext_cnt', 'f_i', 'P']), {}

        r_global = df['rating'].mean()
        r_user_mean = df.groupby('user')['rating'].mean().to_dict()
        r_item_mean = df.groupby('item')['rating'].mean().to_dict()

        # 1) 先在“全体用户”上计算 w，并得到每个 item 的分母 sum_w_all
        df_all = df.copy()
        df_all['w'] = df_all.apply(
            lambda r: self._compute_w(
                r['rating'],
                r_user_mean.get(r['user'], r_global),
                r_item_mean.get(r['item'], r_global),
                r_global
            ),
            axis=1
        )
        sum_w_all = df_all.groupby('item')['w'].sum().to_dict()

        # 2) 仅在嫌疑用户内取 w，并按“全体分母”做归一化得到 w_norm
        df_sus = df_all[df_all['user'].isin(set(sus_users))].copy()
        if df_sus.empty:
            return pd.DataFrame(columns=['item', 'SBA', 'ext_cnt', 'f_i', 'P']), {}

        df_sus['w_norm'] = df_sus.apply(
            lambda r: r['w'] / (sum_w_all.get(r['item'], 1e-8)),
            axis=1
        )
        # SBA_i = 嫌疑用户对 item 的 w_norm 之和（∈[0,1]，表示嫌疑质量占比）
        SBA = df_sus.groupby('item')['w_norm'].sum().to_dict()

        # 3) 极端评分：既支持 push 也支持 nuke（用 tol 做“近极端”容忍）
        tol = cfg.extreme_tolerance
        r_max = cfg.rating_max
        r_min = cfg.rating_min
        df_sus['is_ext'] = ((df_sus['rating'] >= r_max - tol) | (df_sus['rating'] <= r_min + tol)).astype(int)
        ext_cnt = df_sus.groupby('item')['is_ext'].sum().to_dict()

        # 4) f_i：q 自适应（如果配置 q<=0，则用嫌疑人数的 5% 做尺度）
        q = cfg.q if getattr(cfg, 'q', 0) and cfg.q > 0 else max(3, int(math.ceil(0.05 * len(sus_users))))
        f_i = {it: (2.0 / math.pi) * math.atan(cnt / q) for it, cnt in ext_cnt.items()}

        # 5) P_i = SBA_i * f_i
        all_items = set(SBA.keys()) | set(f_i.keys())
        P = {it: SBA.get(it, 0.0) * f_i.get(it, 0.0) for it in all_items}

        stats = pd.DataFrame({
            'item': list(all_items),
            'SBA': [SBA.get(it, 0.0) for it in all_items],
            'ext_cnt': [ext_cnt.get(it, 0) for it in all_items],
            'f_i': [f_i.get(it, 0.0) for it in all_items],
            'P': [P[it] for it in all_items]
        }).sort_values('P', ascending=False).reset_index(drop=True)

        return stats, P

    # ---------- 阶段一 + 阶段二（主计算） ----------
    def irm_tia(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        返回：
          - user_features: 每用户特征（便于可视化与调参）
          - user_scores  : PR/BVL/cluster（便于查看可疑簇）
          - item_stats   : item 的 SBA/ext_cnt/f_i/P 排序表
        """
        # 基础统计
        r_global, r_min, r_max, item_stats, user_stats = self._basic_stats(df)
        # 物品关系：Kulc 和 JS-based sim
        kulc, js_sim = self._pairwise_item_sims(df, r_min, r_max)
        # 构造用户行为特征
        feat_df, X, user_ids = self._user_features(df, r_global, item_stats, user_stats, kulc, js_sim)
        self.last_user_features = feat_df.copy()
        # 可疑用户集合
        sus_users = self._suspicious_users(X, user_ids)
        # 目标物品排序
        item_stats_rank, P = self._rank_items_by_P(df, sus_users)
        self.last_item_stats = item_stats_rank.copy()
        return {
            'user_features': feat_df,
            'user_scores': self.last_user_scores,
            'item_stats': item_stats_rank
        }

    # ---------- 计算 DR_kT ----------
    def compute_dr_kT(
        self,
        item_stats: pd.DataFrame,
        target_item_list: List[int],
        T: Optional[int] = None
    ) -> Dict[str, float]:
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

        # 延迟导入匹配你的项目结构
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

        # (2) 构造合成数据集
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        matrix_factorization = MatrixFactorization(
            n_items=simulator_kwargs.get('n_items', 1682),
            latent_factors=32,
            dataset_name=self.dataset_name,
            device=device
        )
        _ = matrix_factorization.construct_combined_dataset(attack_type=attack_type)

        # (3) 读取合成数据
        combined_path = f'./datasets/{self.dataset_name}/{self.cfg.combined_dataset_filename}'
        df = pd.read_csv(combined_path, sep=' ', header=None, names=['user', 'item', 'rating'])

        # (4) IRM-TIA + (5) DR_kT
        out = self.irm_tia(df)
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
            print(f'----------seed={s + 1} end----------')
            drs = self.run_once(seed=s, simulator_kwargs=simulator_kwargs, attack_type=attack_type)
            per_seed.append({'seed': s, **drs})

        df = pd.DataFrame(per_seed).set_index('seed')
        mean_dict = {k: round(float(df[k].mean()), 4) for k in df.columns}
        std_dict = {k: round(float(df[k].std(ddof=0)), 4) for k in df.columns}

        out = {'mean': mean_dict, 'std': std_dict}
        if return_detail:
            out['per_seed'] = per_seed
        return out
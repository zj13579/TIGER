import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # 不行可换为 ":16:8"

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import numpy as np
import random
import warnings
import time

from GAT import *
from utils.features_construction import FeaturesConstruction
from utils.features_fusion import FeaturesFusion
from identify_abnormal import IdentifyAbnormal

from attack_models.Heuristic_attacks import AttackSimulator
# from attack_models.PGA_attack import AttackSimulator
# from attack_models.SGLD_attack import AttackSimulator
# from attack_models.Infmix_attack import AttackSimulator
# from attack_models.AUSH_attack import AttackSimulator
# from attack_models.TrialAttack import AttackSimulator
# from attack_models.LegUP_attack import AttackSimulator

from attack_models.attack_performance import AttackPerformance
from utils.matrix_factorization import MatrixFactorization
from typing import List, Tuple
from utils.tiger_config import build_config

# 屏蔽这个 FutureWarning
warnings.filterwarnings(
    "ignore",
    message=r".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module=r"sklearn\.utils\.deprecation"
)

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TIGER:
    def __init__(
        self,
        dataset_name = 'ml-100k',
        n_items = 1682,
        num_target_items = 5,
        attack_size = 0.05,
        attack_type = 'random',
        device_str = "cuda:0",
        config=None
    ):
        self.dataset_name = dataset_name
        self.n_items = n_items
        self.num_target_items = num_target_items
        self.attack_size = attack_size
        self.attack_type = attack_type
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.cfg = config

    def run(self, seed: int) -> list:
        c = self.cfg

        # 1) 全局设种子
        set_seed(seed)

        # 2) 生成对抗样本
        simulator = AttackSimulator(
            self.dataset_name,
            self.n_items,
            self.num_target_items,
            self.attack_size,
        )
        target_item_list, fake_profiles = simulator.run(save=True)
        print(f"----------生成对抗样本完成----------\n")

        # 3) MF：商品向量、用户隐向量；构造合成数据集
        matrix_factorization = MatrixFactorization(
            self.n_items,
            c.latent_factors,
            self.dataset_name,
            self.device
        )
        item_vectors = matrix_factorization.obtain_item_vectors(lr_v=c.lr_v, epochs_v=c.epochs_v)
        _, _, _ = matrix_factorization.construct_combined_dataset(attack_type=self.attack_type)
        user_latent_features = matrix_factorization.obtain_user_vectors(lr_u=c.lr_u, epochs_u=c.epochs_u)
        print(f"----------计算商品特征向量、用户隐特征向量部分已完成----------\n")

        # 4) 特征构造，包括局部评分特征、全局评分特征、用户偏好特征
        features_construction = FeaturesConstruction(self.dataset_name, device=self.device, n_global=c.n_global,
                                                     global_kmeans_iter=c.global_kmeans_iter,
                                                     global_kmeans_batch=c.global_kmeans_batch,
                                                     long_user_refine=c.long_user_refine,
                                                     long_user_thresh=c.long_user_thresh,
                                                     user_kmeans_max_k=c.user_kmeans_max_k)
        features_construction.calculate_features(c.similarity_threshold)
        print(f'---------------特征构造部分已完成---------------\n')

        # 5) 特征融合 & 训练GAT
        features_fusion = FeaturesFusion(self.dataset_name)
        user_behavior_features = features_fusion.method_1(c.pca_dimension, rating_features_only=False)
        model = GATs(self.n_items, self.dataset_name, self.device)
        user_representations = model.train_model(
            user_behavior_features, c.num_epochs_gat, c.hidden_dim, c.out_dim, c.lr_gat,
            c.alpha, c.info_th, c.cluster_th, use_latent_features=False
        )
        print(f"----------GAT训练完成，得到用户表征----------\n")

        # 6) 识别异常样本，计算 DR_1T-DR_2T
        iden_abno = IdentifyAbnormal(
                user_representations, self.dataset_name, self.n_items,
                c.lat_fac, c.lr_it, c.num_epochs_it, c.lambda_value_it
        )
        DR_list = iden_abno.method_4(target_item_list, c.w)
        print(f"DR_1T-DR_2T: {DR_list}\n")

        return target_item_list, DR_list


if __name__ == "__main__":
    init_kwargs, config = build_config(
        dataset_name="ml-100k",
        n_items=1682,
        num_target_items=5,
        attack_size=0.05,
        attack_type="random",
        device_str="cuda:0"
    )
    pipeline = TIGER(config=config, **init_kwargs)
    _, DR_list = pipeline.run(seed=0)
<<<<<<< Updated upstream
    print(f"\nDR_1T-DR_2T: {DR_list}")
=======
    print(f"\nDR_1T-DR_2T: {DR_list}")
>>>>>>> Stashed changes

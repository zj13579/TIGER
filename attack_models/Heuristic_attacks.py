import numpy as np
import pandas as pd
import torch
import math
from attack_models.heuristic_utils.heuristic import *
from attack_models.select_target_items import obtain_target_items


class AttackSimulator:
    def __init__(self, dataset_name, n_items, num_target_items=5, attack_size=0.01):
        """
        dataset_path : str   # remained_dataset.txt 路径
        n_items      : int   # 总商品数
        num_target_items  : int # 目标商品的数量
        attack_size  : float # 投毒样本的比例
        save_dir     : str   # 存储虚假画像的目录
        """
        self.dataset_name = dataset_name
        self.dataset_path = f"./datasets/{dataset_name}/remained_dataset.txt"
        self.num_target_items = num_target_items
        self.attack_size = attack_size

        # 读取数据
        self.data = pd.read_csv(self.dataset_path, sep=' ', header=None, names=['user', 'item', 'rating'])
        self.train_kvr = self.data.to_numpy()
        self.num_users = self.data['user'].nunique()
        self.num_items = n_items

    class DummyDataset:
        """模拟 config['dataset'] """
        def __init__(self, train_kvr, n_items):
            self._train_kvr = train_kvr
            self._n_items = n_items

        def info_describe(self):
            return {
                'train_kvr': self._train_kvr,
                'n_items': self._n_items
            }

    def prepare_config(self, seed=None):
        """确定攻击目标并生成攻击配置参数"""
        target_items, _, popular_items, most_popular_items = obtain_target_items(
            self.num_items, self.data, self.dataset_name, self.num_target_items, seed=seed
        )
        print(f"target_items: {target_items}")

        # 用户平均评分数量
        user_rating_counts = self.data.groupby('user').size()
        all_num = math.ceil(user_rating_counts.mean())

        config = {
            'dataset': self.DummyDataset(self.train_kvr, self.num_items),
            'segment_selected_ids': popular_items,
            'bandwagon_selected_ids': most_popular_items,
            'attack_num': math.ceil(self.num_users*self.attack_size),
            'random_filler_num': all_num - 1,
            'average_filler_num': all_num - 1,
            'segment_filler_num': all_num - 6,
            'bandwagon_filler_num': all_num - 11,
            'logging_level': 'INFO'
        }

        return config, target_items

    def run(self, save=False, seed=None):
        """运行所有攻击模型，返回虚假画像字典"""
        config, target_items = self.prepare_config(seed=seed)

        # 四种攻击模型
        attackers = {
            "random": RandomAttack(**config),
            "average": AverageAttack(**config),
            "segment": SegmentAttack(**config),
            "bandwagon": BandwagonAttack(**config)
        }

        fake_profiles = {}
        for name, attacker in attackers.items():
            fake_profiles[name] = attacker.generate_fake(target_id_list=target_items)
            print(f"{name.capitalize()}Attack fake profile shape:", fake_profiles[name].shape)

            if save:
                np.save(f"./results/{self.dataset_name}/{name}_fake_profiles.npy", fake_profiles[name])

        return target_items, fake_profiles

# # ===================== 使用示例 =====================
# if __name__ == "__main__":
#     simulator = AttackSimulator(
#         dataset_path="./datasets/ml-100k/remained_dataset.txt",
#         n_items=1682,  # 商品总数
#         num_target_items=5,
#         save_dir="./results"
#     )
#
#     fake_profiles = simulator.run(save=True)  # 返回 dict，包含四种攻击的虚假画像

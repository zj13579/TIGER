import numpy as np
import pandas as pd
from attack_models.heuristic_utils.utils import  get_logger

class BaseAttacker:
    def __init__(self):
        pass

class RandomAttack(BaseAttacker):
    def __init__(self, **config):
        super(RandomAttack, self).__init__()
        train_kv_array = config['dataset'].info_describe()['train_kvr']
        self.n_items = config['dataset'].info_describe()['n_items']
        self.global_mean = np.mean(train_kv_array[:, 2])
        self.global_std = np.std(train_kv_array[:, 2])
        self.attack_num = config['attack_num']
        self.filler_num = config['random_filler_num']
        self.logger = get_logger(__name__, level=config['logging_level'])

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        num = int(self.attack_num / len(target_id_list))
        self.logger.debug('每个目标商品分配 %s 个攻击用户' % num)

        # 设置目标评分为5
        for i in range(self.attack_num):
            # 使用取余保证循环分配目标商品
            target_item = target_id_list[i % len(target_id_list)]
            fake_profiles[i, target_item] = 5

        # for i in range(len(target_id_list)):
        #     fake_profiles[i * num : (i + 1) * num, target_id_list[i]] = 5  # 给第i批次的虚假用户，在第i个目标商品上的评分设置为5

        # 填充 filler 项
        filler_pool = list(set(range(self.n_items)) - set(target_id_list))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)

        sampled_cols = np.reshape(
            np.array([
                filler_sampler([filler_pool, self.filler_num])
                for _ in range(self.attack_num)
            ]),
            (-1)
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]

        sampled_values = np.random.normal(
            loc=self.global_mean,
            scale=self.global_std,
            size=(self.attack_num * self.filler_num)
        )
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1

        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        return fake_profiles



class AverageAttack(BaseAttacker):
    def __init__(self, **config):
        super().__init__()
        kvr = config['dataset'].info_describe()['train_kvr']
        self.n_items = config['dataset'].info_describe()['n_items']

        self.global_mean = np.mean(kvr[:, 2])
        self.global_std = np.std(kvr[:, 2])

        self.item_mean_dict = {}
        self.item_std_dict = {}
        for iid in np.unique(kvr[:, 1]):
            self.item_mean_dict[iid] = np.mean(kvr[kvr[:, 1] == iid][:, 2])
            self.item_std_dict[iid] = np.std(kvr[kvr[:, 1] == iid][:, 2])
        self.attack_num = config['attack_num']
        self.filler_num = config['average_filler_num']
        self.logger = get_logger(__name__, level=config['logging_level'])

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        num = int(self.attack_num / len(target_id_list))

        for i in range(self.attack_num):
            target_item = target_id_list[i % len(target_id_list)]
            fake_profiles[i, target_item] = 5

        filler_pool = list(set(range(self.n_items)) - set(target_id_list))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)

        sampled_cols = np.reshape(
            np.array([
                filler_sampler([filler_pool, self.filler_num])
                for _ in range(self.attack_num)
            ]),
            (-1)
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]

        sampled_values = [
            np.random.normal(
                loc=self.item_mean_dict.get(iid, self.global_mean),
                scale=self.item_std_dict.get(iid, self.global_std)
            )
            for iid in sampled_cols
        ]
        sampled_values = np.round(sampled_values)
        sampled_values = np.clip(sampled_values, 1, 5)

        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        return fake_profiles



class SegmentAttack(BaseAttacker):
    def __init__(self, **config):
        super().__init__()
        kvr = config['dataset'].info_describe()['train_kvr']
        self.n_items = config['dataset'].info_describe()['n_items']
        self.selected_ids = config['segment_selected_ids']
        self.attack_num = config['attack_num']
        self.filler_num = config['segment_filler_num']
        self.logger = get_logger(__name__, level=config['logging_level'])

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)

        # Step 1：打目标商品评分
        num = int(self.attack_num / len(target_id_list))
        for i in range(self.attack_num):
            target_item = target_id_list[i % len(target_id_list)]
            fake_profiles[i, target_item] = 5

        # Step 2：对 selected_ids 全部打 5
        fake_profiles[:, self.selected_ids] = 5

        # Step 3：从剩余商品中采样 fillers，打 1 分
        filler_pool = list(
            set(range(self.n_items)) - set(target_id_list) - set(self.selected_ids)
        )
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([
                filler_sampler([filler_pool, self.filler_num])
                for _ in range(self.attack_num)
            ]),
            (-1)
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]
        sampled_values = np.ones_like(sampled_rows)
        fake_profiles[sampled_rows, sampled_cols] = sampled_values

        return fake_profiles



class BandwagonAttack(BaseAttacker):
    def __init__(self, **config):
        super().__init__()
        kvr = config['dataset'].info_describe()['train_kvr']
        self.global_mean = np.mean(kvr[:, 2])
        self.global_std = np.std(kvr[:, 2])
        train_data_df = pd.DataFrame(kvr, columns=['user_id', 'item_id', 'rating'])
        self.n_items = config['dataset'].info_describe()['n_items']

        self.selected_ids = config['bandwagon_selected_ids']
        self.attack_num = config['attack_num']
        self.filler_num = config['bandwagon_filler_num']

        if len(self.selected_ids) == 0:
            sorted_item_pop_df = (
                train_data_df.groupby('item_id')
                .agg('count')
                .sort_values('user_id')
                .index[::-1]
            )
            self.selected_ids = sorted_item_pop_df[:11].to_list()

        self.logger = get_logger(__name__, level=config['logging_level'])

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)

        # Step 1: 填充目标商品评分为 5
        num = int(self.attack_num / len(target_id_list))
        for i in range(self.attack_num):
            target_item = target_id_list[i % len(target_id_list)]
            fake_profiles[i, target_item] = 5

        # Step 2: 填充 selected_ids 为 5
        fake_profiles[:, self.selected_ids] = 5

        # Step 3: 填充 filler 项（打随机评分）
        # 1) 计算可选填充池
        filler_pool = np.array(list(set(range(self.n_items)) - set(target_id_list) - set(self.selected_ids)))
        pool_size = int(filler_pool.size)

        # 2) 规整 filler_num：支持整数或比例(0,1]
        k = self.filler_num
        # 若是比例，按候选池大小换算
        if isinstance(k, float) and 0 < k <= 1:
            k = int(round(k * pool_size))

        # 强制为非负整数
        try:
            k = int(k)
        except Exception:
            raise ValueError(f"bandwagon_filler_num 无法转换为整数: {self.filler_num}")

        # 边界保护：负数置0；超过池大小则截断或改为放回采样（二选一）
        if k < 0:
            k = 0
        if pool_size == 0 or k == 0:
            # 没有可填充项，直接返回当前 fake_profiles
            return fake_profiles

        # 如果你希望绝不重复物品，采用截断到池大小：
        k_eff = min(k, pool_size)
        # 如果你更希望严格使用 k 个且允许重复，可改成：
        # replace_flag = (k > pool_size); k_eff = k

        # 3) 为每个攻击样本抽取 k_eff 个 filler 物品
        sampled_cols = np.concatenate([
            np.random.choice(filler_pool, size=k_eff, replace=False)  # 如上改为 replace=replace_flag
            for _ in range(self.attack_num)
        ])
        sampled_rows = np.repeat(np.arange(self.attack_num), k_eff)

        # 4) 生成对应评分
        sampled_values = np.random.normal(
            loc=self.global_mean,
            scale=self.global_std,
            size=(self.attack_num * k_eff)
        )
        sampled_values = np.round(sampled_values)
        sampled_values = np.clip(sampled_values, 1, 5)

        fake_profiles[sampled_rows, sampled_cols] = sampled_values

        return fake_profiles

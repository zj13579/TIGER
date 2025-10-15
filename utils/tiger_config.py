from types import SimpleNamespace

def _get_dataset_defaults(dataset_name: str) -> dict:
    if dataset_name == 'ml-100k':
        return dict(
            # 计算商品特征向量、用户隐特征向量
            latent_factors=32,
            lr_v=0.05,
            epochs_v=5000,
            lr_u=0.1,
            epochs_u=2000,
            # 特征构造
            n_global=128,
            global_kmeans_iter=50,
            global_kmeans_batch=512,
            long_user_refine=False,
            long_user_thresh=250,
            user_kmeans_max_k=5,
            similarity_threshold=0.25,
            # 特征融合
            pca_dimension=16,
            # GAT
            num_epochs_gat=500,
            hidden_dim=64,
            out_dim=32,
            info_th=1,      # 默认为1
            cluster_th=1,   # 默认为1
            lr_gat=0.005,
            alpha=0.2,
            # 识别目标商品
            lat_fac=64,
            lr_it=0.1,
            num_epochs_it=300,
            lambda_value_it=0.001,
            # 识别异常样本
            w=[0.6, 0.2, 0.2],
        )
    elif dataset_name == 'ml-1m':
        return dict(
            # 计算商品特征向量、用户隐特征向量
            latent_factors=64,
            lr_v=0.05,
            epochs_v=3000,
            lr_u=0.05,
            epochs_u=1000,
            # 特征构造
            n_global=256,                 # 推荐 256~512
            global_kmeans_iter=100,
            global_kmeans_batch=2048,
            long_user_refine=False,
            long_user_thresh=400,         # “超长用户”阈值
            user_kmeans_max_k=7,          # 在 6–8 之间微调
            similarity_threshold=0.30,    # 在 0.28–0.35 区间微调
            # 特征融合
            pca_dimension=32,
            # GAT
            num_epochs_gat=150,
            hidden_dim=128,
            out_dim=64,
            info_th=1,     # 默认为1
            cluster_th=1,  # 默认为1
            lr_gat=0.001,
            alpha=0.2,
            # 识别目标商品
            lat_fac=64,
            lr_it=0.05,
            num_epochs_it=200,
            lambda_value_it=0.001,
            # 识别异常样本
            w=[0.6, 0.2, 0.2],
        )
    elif dataset_name == 'automotive':
        return dict(
            # 数据集划分
            min_sparsity=0.996,
            max_sparsity=0.9972,
            # 攻击效果验证（可选）
            lat_factors=32,
            lr=0.06,
            num_epochs=300,
            lambda_value=0.001,
            # 计算商品特征向量、用户隐特征向量
            latent_factors=32,
            lr_v=0.06,
            epochs_v=3000,
            lr_u=0.06,
            epochs_u=1500,
            # 特征构造
            n_global=96,
            global_kmeans_iter=50,
            global_kmeans_batch=512,
            long_user_refine=False,
            long_user_thresh=200,         # (把上面开为True时才生效)  # “超长用户”阈值
            user_kmeans_max_k=4,
            similarity_threshold=0.30,    # 在 0.28–0.35 区间微调
            # 特征融合
            pca_dimension=16,
            # GAT
            num_epochs_gat=300,
            hidden_dim=64,
            out_dim=32,
            info_th=1,      # 默认为1
            cluster_th=1,   # 默认为1
            lr_gat=0.005,
            alpha=0.2,
            # 识别目标商品
            lat_fac=32,
            lr_it=0.06,
            num_epochs_it=300,
            lambda_value_it=0.001,
            # 识别异常样本
            w=[0.6, 0.2, 0.2],
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

def build_config(
    dataset_name: str,
    *,
    # TIGER 基本参数
    n_items: int = 1682,
    num_target_items: int = 5,
    attack_size: float = 0.05,
    attack_type: str = 'random',
    device_str: str = "cuda:0",
):
    init_kwargs = dict(
        dataset_name=dataset_name,
        n_items = n_items,
        num_target_items=num_target_items,
        attack_size=attack_size,
        attack_type=attack_type,
        device_str=device_str,
    )

    defaults = _get_dataset_defaults(dataset_name)
    cfg = defaults.copy()  # 其余超参作为 cfg

    return init_kwargs, SimpleNamespace(**cfg)
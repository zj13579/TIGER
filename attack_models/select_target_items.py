import pandas as pd
import random

# random.seed(0)
def obtain_target_items(num_items, data, dataset_name, num_target_items, seed=None):

    if seed is not None:
        random.seed(seed)

    # 设定商品总数
    all_item_ids = set(range(num_items))

    # 统计评分频次（仅包含出现过的商品）
    rated_counts = data['item'].value_counts().sort_index()  # Series: item_id -> count

    # 转为 DataFrame
    rated_stats = rated_counts.reset_index()
    rated_stats.columns = ['item_id', 'rating_count']
    most_popular_items = rated_stats.sort_values('rating_count', ascending=False).head(10)['item_id'].tolist()

    # 找出未出现的商品，补充为 0 次
    missing_item_ids = all_item_ids - set(rated_stats['item_id'])
    missing_df = pd.DataFrame({
        'item_id': list(missing_item_ids),
        'rating_count': 0
    })

    # 合并所有商品信息
    full_stats = pd.concat([rated_stats, missing_df], ignore_index=True).sort_values('item_id').reset_index(drop=True)

    # 设置评分数分桶规则

    if dataset_name == "ml-100k":
        bins = [0, 10, 200, float('inf')]
        labels = ['0-9', '10-199', '200+']
    elif dataset_name == "ml-1m":
        bins = [0, 10, 20, 50, 300, 600, 900, float('inf')]
        labels = ['0-9', '10-19', '20-49', '50-299', '300-599', '600-899', '900+']
    elif dataset_name == "automotive":
        bins = [0, 5, 10, 30, float('inf')]
        labels = ['0-4', '5-9', '10-29', '30+']

    # 分桶
    full_stats['bucket'] = pd.cut(full_stats['rating_count'], bins=bins, labels=labels, right=False)

    # 输出每个桶的商品ID列表
    bucket_dict = {}
    for label in labels:
        bucket_items = full_stats[full_stats['bucket'] == label]['item_id'].tolist()
        bucket_dict[label] = bucket_items
        # print(f"区间 {label}：{len(bucket_items)} 件商品")
        # print(f"示例ID（前10个）: {bucket_items[:10]}")
        # print('---')

    if dataset_name == "ml-100k":
        unpopular_items_list = bucket_dict['0-9']  # 数量：210+510个（210未评分商品）
        random_items_list = bucket_dict['10-199']
        popular_items_list = bucket_dict['200+']
    elif dataset_name == "ml-1m":
        unpopular_items_list = bucket_dict['0-9']
        random_items_list = bucket_dict['50-299']
        popular_items_list = bucket_dict['900+']
    elif dataset_name == "automotive":
        unpopular_items_list = bucket_dict['0-4']
        random_items_list = bucket_dict['5-9']
        popular_items_list = bucket_dict['30+']

    # 从列表中随机抽取一定数量不重复的元素
    unpopular_items = random.sample(unpopular_items_list, k=num_target_items)
    random_items = random.sample(random_items_list, k=num_target_items)
    popular_items = random.sample(popular_items_list, k=num_target_items)  # 从流行商品中随机选择5件商品作为’selected_items‘

    return unpopular_items, random_items, popular_items, most_popular_items

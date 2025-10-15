import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1.读入数据
############
def read_data(num_users, num_items, path):
    user_ids = []
    item_ids = []
    rating_matrix = np.zeros((num_users, num_items))
    with open(path, "r") as file:
        for line in file:
            user, item, rating = map(str, line.strip().split())
            user = int(user)
            item = int(item)
            rating = float(rating)

            user_ids.append(user)
            item_ids.append(item)

            # 将评分填充到对应的位置（索引从0开始）
            rating_matrix[user, item] = rating

    user_ids = torch.tensor(user_ids)
    item_ids = torch.tensor(item_ids)

    # 将数据转换为PyTorch张量
    rating_matrix = torch.from_numpy(rating_matrix)

    return user_ids, item_ids, rating_matrix



# 2.矩阵分解
############
'''
对真实数据集的矩阵分解和对未知数据集的矩阵分解
'''
class MF(nn.Module):
    def __init__(self, num_users, num_items, latent_factors, device, item_vectors=None):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_factors = latent_factors
        self.device = device
        self.user_indices = torch.arange(num_users).to(device)
        self.item_indices = torch.arange(num_items).to(device)

        self.user_embeddings = nn.Embedding(num_users, latent_factors)

        # 商品向量是否固定
        self.item_embeddings = nn.Embedding(num_items, latent_factors)
        if item_vectors is not None:
            self.item_embeddings.weight = nn.Parameter(item_vectors.to(device), requires_grad=False)

        # ✅ 保存标志，方便 train_model 使用
        self.item_vectors_fixed = item_vectors is not None

    def forward(self):
        user_latent = self.user_embeddings(self.user_indices)
        item_latent = self.item_embeddings(self.item_indices)
        ratings_pred = torch.matmul(user_latent, item_latent.t())
        return user_latent, item_latent, ratings_pred

    def train_model(self, path, lr, num_epochs):
        user_train_ids, item_train_ids, train_data = read_data(self.num_users, self.num_items, path=path)

        self.to(self.device)
        train_data = train_data.to(self.device)
        criterion = nn.MSELoss().to(self.device)

        # ✅ 根据 item_vectors 是否固定来决定优化器
        if self.item_vectors_fixed:
            optimizer = optim.Adam(self.user_embeddings.parameters(), lr)  # 只更新用户向量
        else:
            optimizer = optim.Adam(self.parameters(), lr)  # 同时更新用户和物品向量

        train_ratings = train_data[user_train_ids, item_train_ids].to(torch.float32)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            _, _, ratings_pred = self()
            train_ratings_pred = ratings_pred[user_train_ids, item_train_ids].to(torch.float32)
            train_loss = criterion(train_ratings_pred, train_ratings)
            train_loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], train_Loss: {train_loss.item()}")



# 3.用户特征融合部分的函数
########################
# === 1. 定义一个AutoEncoder降维模块
class FeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 可调
            nn.ReLU(),
            nn.Linear(32, 16),  # 输出降维为16，可自动调参
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# === 2. 训练AutoEncoder并获得用户降维特征
def train_autoencoder(user_feature_dict, num_epochs=100, lr=1e-3):
    user_ids = list(user_feature_dict.keys())
    feature_matrix = [user_feature_dict[uid] for uid in user_ids]
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

    model = FeatureAutoEncoder(input_dim=feature_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        encoded, decoded = model(feature_tensor)
        loss = criterion(decoded, feature_tensor)
        loss.backward()
        optimizer.step()

    # 获取降维结果
    with torch.no_grad():
        reduced_features, _ = model(feature_tensor)
        reduced_dict = {uid: reduced_features[i].numpy() for i, uid in enumerate(user_ids)}
    return reduced_dict

# === 3. 注意力机制进行特征融合
class FeatureAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureAttentionFusion, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.softmax(Q @ K.transpose(0, 1) / (Q.size(-1) ** 0.5), dim=1)
        context = scores @ V
        out = self.out(context)
        return out



# 4.top-K推荐
############
# 得到未评分商品的坐标列表
def obtain_filtered_indices(user_ids, item_ids, defaultdict, num_items):

    ratings_indices = torch.stack((user_ids, item_ids), dim=1)
    ratings_indices_list = ratings_indices.tolist()

    ratings_indices_dict = defaultdict(list)
    for user_ids, item_ids in ratings_indices_list:
        ratings_indices_dict[user_ids].append(item_ids)

    # 创建每一个sublist元素的补集,最终得到针对每个用户所有未评分商品id的集合
    filtered_list = []
    full_set = set(range(num_items))
    for i in range(len(ratings_indices_dict)):
        sub_list = ratings_indices_dict[i]
        complement_set = full_set - set(sub_list)  # 计算补集
        complement_list = list(complement_set)
        filtered_list.append(complement_list)

    # 构造未评分商品的坐标列表
    filtered_indices = []
    for i in range(len(filtered_list)):
        result = [(i, j) for j in filtered_list[i]]
        filtered_indices.append(result)
    filtered_indices = [item for sublist in filtered_indices for item in sublist]

    return ratings_indices_dict, filtered_indices

# 获取每个用户的商品推荐列表
def obtain_rec_list(ratings_pred, filtered_indices, defaultdict, num_users, ratings_indices_dict, K=10):
    filtered_ratings_list = [(item[0], item[1], (ratings_pred[item[0], item[1]]).item()) for item in filtered_indices]

    user_ratings_dict = defaultdict(list)
    for user_id, item_id, rating in filtered_ratings_list:
        user_ratings_dict[user_id].append((item_id, rating))

    # 同时包含item-id与rating的前10件商品的推荐列表
    Top_K_list = []
    for i in range(num_users):
        sorted_result = sorted(user_ratings_dict[i], key=lambda x: x[1], reverse=True)
        sorted_top_K = sorted_result[:K]
        Top_K_list.append(sorted_top_K)

    Top_K_item_list = []
    for i in range(num_users):
        sublist = []
        for j in range(K):
            sublist.append(Top_K_list[i][j][0])

        assert all(item not in ratings_indices_dict[i] for item in
                   sublist), "每件商品都应该为未评分商品"
        Top_K_item_list.append(sublist)

    # 同时包含item-id与rating的所有商品的推荐列表
    rec_list = []
    for i in range(num_users):
        sorted_result = sorted(user_ratings_dict[i], key=lambda x: x[1], reverse=True)
        rec_list.append(sorted_result)
    # 只包含item_id的所有商品的推荐列表
    rec_item_list = []
    for i in range(num_users):
        sublist = []
        length = len(rec_list[i])
        for j in range(length):
            sublist.append(rec_list[i][j][0])

        assert all(item not in ratings_indices_dict[i] for item in
                   sublist), "每件商品都应该为未评分商品"
        rec_item_list.append(sublist)

    return Top_K_item_list, rec_item_list
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from utils.functions import read_data

class LFM(nn.Module):
    def __init__(self, num_users, num_items, latent_factors, device):
        super(LFM, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_factors = latent_factors
        self.device = device
        self.user_indices = torch.arange(num_users).to(device)
        self.item_indices = torch.arange(num_items).to(device)

        self.user_embeddings = nn.Embedding(num_users, latent_factors)
        self.item_embeddings = nn.Embedding(num_items, latent_factors)

    def forward(self, lambda_value):
        user_latent = self.user_embeddings(self.user_indices)
        item_latent = self.item_embeddings(self.item_indices)
        ratings_pred = torch.matmul(user_latent, item_latent.t())
        user_latent_l2_norms = (torch.norm(user_latent, dim=1)) ** 2
        sum_of_user_l2_norms = torch.sum(user_latent_l2_norms)
        item_latent_l2_norms = (torch.norm(item_latent, dim=1)) ** 2
        sum_of_item_l2_norms = torch.sum(item_latent_l2_norms)
        loss_regularization = lambda_value * (sum_of_user_l2_norms + sum_of_item_l2_norms)

        return user_latent, item_latent, ratings_pred, loss_regularization


    def train_model(self, path, lr, num_epochs, lambda_value):
        user_train_ids, item_train_ids, train_data = read_data(self.num_users, self.num_items, path=path)  # 读入训练集上的数据

        self.to(self.device)  # 创建模型实例
        train_data = train_data.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr)

        # 模型训练
        train_ratings = train_data[user_train_ids, item_train_ids]  # 获取训练集上相应位置的真实评分

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            _, _, ratings_pred, loss_regularization = self(lambda_value)  # 得到这批样本的评分预测结果和正则化项损失
            train_ratings_pred = ratings_pred[user_train_ids, item_train_ids].to(torch.float64)  # 获取相应位置上的预测评分
            train_loss = criterion(train_ratings_pred, train_ratings) + loss_regularization
            train_loss.backward()
            optimizer.step()

            # 打印训练效果
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], train_Loss: {train_loss.item()}")


    def retrain_model(self, poisoned_train_data, lr, num_epochs, lambda_value):

        nonzero_indices = torch.nonzero(poisoned_train_data)
        user_train_ids = nonzero_indices[:, 0]
        item_train_ids = nonzero_indices[:, 1]

        self.to(self.device)  # 创建模型实例
        poisoned_train_data = poisoned_train_data.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr)

        # 模型训练
        train_ratings = poisoned_train_data[user_train_ids, item_train_ids]  # 获取训练集上相应位置的真实评分

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            _, _, ratings_pred, loss_regularization = self(lambda_value)  # 得到这批样本的评分预测结果和正则化项损失
            train_ratings_pred = ratings_pred[user_train_ids, item_train_ids].to(torch.float64)  # 获取相应位置上的预测评分
            train_loss = criterion(train_ratings_pred, train_ratings) + loss_regularization
            train_loss.backward()
            optimizer.step()

            # 打印训练效果
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], train_Loss: {train_loss.item()}")
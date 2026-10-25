"""用于评估 GPT-2 文本增强收益的纯 ID NCF 与热门电影基线。"""

import torch
import torch.nn as nn


def _valid_item_mask(n_items, valid_item_ids):
    """电影目录决定候选集合；电影 ID 的空洞不应进入推荐列表。"""
    item_ids = torch.as_tensor(list(valid_item_ids), dtype=torch.long)
    if ((item_ids < 0) | (item_ids >= n_items)).any():
        raise ValueError("valid_item_ids 中的电影 ID 必须在 [0, n_items) 内")
    mask = torch.zeros(n_items, dtype=torch.bool)
    mask[item_ids] = True
    return mask


class NCFRecommender(nn.Module):
    """不读取文本特征的 NCF；valid_item_ids 应传 dataset.items_df.index。"""

    def __init__(self, n_users, n_items, valid_item_ids, embed_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )
        self.register_buffer(
            "valid_item_mask", _valid_item_mask(n_items, valid_item_ids)
        )
        self.n_items = n_items

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    @property
    def device(self):
        return self.user_embedding.weight.device

    def score(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        return self.mlp(torch.cat([user_vec, item_vec], dim=-1)).squeeze(-1)

    def forward(self, batch_data):
        user_ids, positive_item_ids, negative_item_ids = batch_data
        return self.score(user_ids, positive_item_ids), self.score(
            user_ids, negative_item_ids
        )

    def calculate_loss(self, positive_scores, negative_scores):
        """与 HybridRecommender 使用完全一致的 BPR 损失。"""
        delta = positive_scores - negative_scores
        return -torch.mean(torch.log(torch.sigmoid(delta) + 1e-8))

    @torch.no_grad()
    def full_sort_predict(self, user_ids):
        user_ids = user_ids.to(self.device)
        batch_size = len(user_ids)
        all_item_ids = torch.arange(self.n_items, device=self.device)
        users = user_ids[:, None].expand(batch_size, self.n_items).reshape(-1)
        items = all_item_ids[None, :].expand(batch_size, self.n_items).reshape(-1)
        scores = self.score(users, items).view(batch_size, self.n_items)
        scores[:, ~self.valid_item_mask] = -torch.inf
        return scores


class PopularityRecommender(nn.Module):
    """仅按训练集交互次数排序，不读取验证/测试交互或文本特征。"""

    def __init__(self, dataset):
        super().__init__()
        self.n_items = dataset.get_item_num()
        self.register_buffer(
            "valid_item_mask",
            _valid_item_mask(self.n_items, dataset.items_df.index),
        )
        train_items = torch.as_tensor(
            dataset.get_split_data("train")["movie_id"].to_numpy(),
            dtype=torch.long,
        )
        if ((train_items < 0) | (train_items >= self.n_items)).any():
            raise ValueError("训练集中的电影 ID 必须在 [0, n_items) 内")
        self.register_buffer(
            "item_popularity",
            torch.bincount(train_items, minlength=self.n_items).float(),
        )

    @property
    def device(self):
        return self.item_popularity.device

    @torch.no_grad()
    def full_sort_predict(self, user_ids):
        # clone 确保评测器屏蔽某用户历史时，不修改其他用户或持久计数。
        scores = self.item_popularity[None, :].expand(len(user_ids), -1).clone()
        scores[:, ~self.valid_item_mask] = -torch.inf
        return scores

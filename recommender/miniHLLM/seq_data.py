"""按时间排序的用户序列与 leave-one-out 划分。

每个用户的评分按时间戳排序得到序列 [i_1, ..., i_n]：
- 训练：输入 [i_1, ..., i_{n-3}]，目标 [i_2, ..., i_{n-2}]（每个位置预测下一部）；
- 验证：输入 [i_1, ..., i_{n-2}]，目标 i_{n-1}；
- 测试：输入 [i_1, ..., i_{n-1}]，目标 i_n。
序列只保留最近 max_len 个，右侧用 0 填充；0 不是任何电影的编号。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


class SeqData:
    def __init__(self, data_path, max_len=50):
        data_path = Path(data_path)
        ratings = pd.read_csv(
            data_path / "ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )
        self.movies = pd.read_csv(
            data_path / "movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        ).set_index("movie_id")

        self.max_len = max_len
        self.n_items = int(self.movies.index.max()) + 1
        self.n_users = int(ratings["user_id"].max()) + 1

        # 稳定排序：同一时间戳保持文件中的原始顺序。
        ratings = ratings.sort_values(["user_id", "timestamp"], kind="stable")
        grouped = ratings.groupby("user_id")["movie_id"].apply(list)
        self.user_ids = torch.as_tensor(grouped.index.to_numpy(), dtype=torch.long)
        self.sequences = grouped.to_list()

        # 有效电影：movies.dat 中出现的编号；编号空洞与 0 号填充位不能进入排序。
        self.valid_item_mask = torch.zeros(self.n_items, dtype=torch.bool)
        self.valid_item_mask[self.movies.index.to_numpy()] = True

        # 电影热度只统计训练部分，用于 Popularity 基线和冷启动分桶。
        train_items = np.concatenate([np.asarray(s[:-2]) for s in self.sequences])
        self.item_pop = torch.bincount(
            torch.as_tensor(train_items, dtype=torch.long), minlength=self.n_items
        )
        self._build()

    def _pad(self, seqs, fill=0):
        out = torch.full((len(seqs), self.max_len), fill, dtype=torch.long)
        for row, s in enumerate(seqs):
            s = s[-self.max_len:]
            out[row, : len(s)] = torch.as_tensor(s, dtype=torch.long)
        return out

    def _history_mask(self, seqs):
        mask = torch.zeros(len(seqs), self.n_items, dtype=torch.bool)
        for row, s in enumerate(seqs):
            mask[row, torch.as_tensor(s, dtype=torch.long)] = True
        return mask

    def _build(self):
        seqs = self.sequences
        self.train_seq = self._pad([s[:-3] for s in seqs])
        self.train_target = self._pad([s[1:-2] for s in seqs], fill=-100)

        self.valid_seq = self._pad([s[:-2] for s in seqs])
        self.valid_target = torch.as_tensor([s[-2] for s in seqs], dtype=torch.long)
        self.valid_history = self._history_mask([s[:-2] for s in seqs])

        self.test_seq = self._pad([s[:-1] for s in seqs])
        self.test_target = torch.as_tensor([s[-1] for s in seqs], dtype=torch.long)
        self.test_history = self._history_mask([s[:-1] for s in seqs])

    def title(self, movie_id):
        return self.movies.loc[int(movie_id), "title"]

    def __str__(self):
        lengths = [len(s) for s in self.sequences]
        return (
            f"SeqData: users={len(self.sequences)}, items={int(self.valid_item_mask.sum())}, "
            f"interactions={sum(lengths)}, seq_len min/median/max="
            f"{min(lengths)}/{int(np.median(lengths))}/{max(lengths)}, max_len={self.max_len}"
        )


if __name__ == "__main__":
    data = SeqData(Path(__file__).resolve().parents[2] / "ml-1m")
    print(data)
    print("train_seq", tuple(data.train_seq.shape), "valid_seq", tuple(data.valid_seq.shape))
    pop = data.item_pop[data.test_target]
    for lo, hi in [(0, 20), (20, 200), (200, 10**9)]:
        n = int(((pop >= lo) & (pop < hi)).sum())
        print(f"test targets with train popularity in [{lo}, {hi}): {n}")

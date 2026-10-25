import unittest

import pandas as pd
import torch

from baselines import NCFRecommender, PopularityRecommender
from recommender import HybridRecommender


class ToyDataset:
    """电影 0、2、4 是 ID 空洞；电影 5 只在验证和测试集中出现。"""

    def __init__(self):
        self.items_df = pd.DataFrame(index=[1, 3, 5])
        self.interactions_df = pd.DataFrame(
            {
                "movie_id": [1, 1, 3, 5, 5, 5, 5],
                "split": [0, 0, 0, 1, 1, 2, 2],
            }
        )
        self.requested_splits = []

    def get_item_num(self):
        return 6

    def get_split_data(self, split):
        self.requested_splits.append(split)
        split_ids = {"train": 0, "validation": 1, "test": 2}
        return self.interactions_df[
            self.interactions_df["split"] == split_ids[split]
        ].copy()


class NCFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dataset = ToyDataset()
        self.model = NCFRecommender(
            4, self.dataset.get_item_num(), self.dataset.items_df.index
        )

    def test_id_only_model_keeps_catalog_items_and_masks_id_holes(self):
        # 模型的构造与预测都无需文本文件；测试集独有的有效电影仍是候选。
        scores = self.model.full_sort_predict(torch.tensor([1, 2]))
        self.assertEqual(scores.shape, (2, 6))
        self.assertTrue(torch.isfinite(scores[:, [1, 3, 5]]).all())
        self.assertTrue(torch.isneginf(scores[:, [0, 2, 4]]).all())
        self.assertFalse(hasattr(self.model, "text_features"))
        self.assertFalse(hasattr(self.model, "text_projector"))
        self.assertEqual(self.model.mlp[0].in_features, 64)
        self.assertEqual(self.model.mlp[0].out_features, 64)

    def test_single_example_keeps_batch_dimension_and_matches_full_sort(self):
        batch = torch.tensor([[1], [3], [5]])
        positive, negative = self.model(batch)
        self.assertEqual(positive.shape, (1,))
        self.assertEqual(negative.shape, (1,))
        all_scores = self.model.full_sort_predict(batch[0])
        self.assertEqual(all_scores.shape, (1, 6))
        torch.testing.assert_close(all_scores[:, 3], positive)
        torch.testing.assert_close(all_scores[:, 5], negative)

    def test_bpr_matches_hybrid_and_backpropagates_to_both_embeddings(self):
        batch = torch.tensor([[1, 2], [1, 3], [5, 1]])
        positive, negative = self.model(batch)
        loss = self.model.calculate_loss(positive, negative)
        expected = HybridRecommender.calculate_loss(None, positive, negative)
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (
            self.model.user_embedding.weight,
            self.model.item_embedding.weight,
            self.model.mlp[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


class PopularityTests(unittest.TestCase):
    def setUp(self):
        self.dataset = ToyDataset()
        self.model = PopularityRecommender(self.dataset)

    def test_counts_only_train_and_keeps_unseen_valid_items(self):
        self.assertEqual(self.dataset.requested_splits, ["train"])
        self.model.eval().to("cpu")
        self.assertEqual(self.model.device, torch.device("cpu"))
        self.assertEqual(list(self.model.parameters()), [])
        scores = self.model.full_sort_predict(torch.tensor([1]))
        self.assertEqual(scores.shape, (1, 6))
        torch.testing.assert_close(scores[0, [1, 3, 5]], torch.tensor([2., 1., 0.]))
        self.assertTrue(torch.isneginf(scores[:, [0, 2, 4]]).all())
        self.assertEqual(scores.argmax(dim=1).item(), 1)

    def test_history_masking_does_not_modify_other_users_or_counts(self):
        scores = self.model.full_sort_predict(torch.tensor([1, 2]))
        scores[0, 1] = -torch.inf
        self.assertEqual(scores[1, 1].item(), 2)
        fresh_scores = self.model.full_sort_predict(torch.tensor([1]))
        self.assertEqual(fresh_scores[0, 1].item(), 2)

    def test_held_out_changes_cannot_change_popularity(self):
        before = self.model.full_sort_predict(torch.tensor([1]))
        held_out = self.dataset.interactions_df["split"] != 0
        self.dataset.interactions_df.loc[held_out, "movie_id"] = 3
        after = PopularityRecommender(self.dataset).full_sort_predict(torch.tensor([1]))
        torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()

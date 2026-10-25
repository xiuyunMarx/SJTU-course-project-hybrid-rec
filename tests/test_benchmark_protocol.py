"""协议级验证：只检查小样本排序、采样与汇总，不运行模型训练。"""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from baselines import NCFRecommender
from dataloader import EvalDataLoader, TrainDataLoader
from recommender import HybridRecommender
from run_benchmarks import (
    EvaluationOnly,
    save_summary,
    seed_model,
    seed_shuffle,
    tensor_hash,
)


class FixedScores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 各用户训练历史具有最高原始分数，确保漏掉历史屏蔽会改变答案。
        self.register_buffer("scores", torch.tensor([
            [0., 0., 0., 0., 0., 0.],
            [-10., 100., 9., 8., 7., 6.],
            [-10., 9., 100., 6., 8., 7.],
        ]))

    def full_sort_predict(self, users):
        return self.scores[users].clone()


class EvaluationProtocolTests(unittest.TestCase):
    def test_original_metrics_match_hand_calculation_after_history_mask(self):
        train = pd.DataFrame({"user_id": [1, 2], "movie_id": [1, 2]})
        held_out = pd.DataFrame({"user_id": [1, 1, 2], "movie_id": [3, 4, 1]})
        loader = EvalDataLoader(held_out, train, batch_size=1, device="cpu")
        model = FixedScores()
        result = EvaluationOnly(model, "cpu")._evaluate(loader, topk=(1, 3))

        # 屏蔽后，用户1排 [2,3,4]、用户2排 [1,4,5]。
        # 用户1的两个答案分别处于第2/3位；用户2唯一答案处于第1位。
        first_user_ndcg3 = (1 / math.log2(3) + 1 / math.log2(4)) / (
            1 + 1 / math.log2(3)
        )
        self.assertEqual(result["HR@1"], 0.5)
        self.assertEqual(result["HR@3"], 1.0)
        self.assertEqual(result["NDCG@1"], 0.5)
        self.assertAlmostEqual(result["NDCG@3"], (first_user_ndcg3 + 1) / 2)
        # 评测器修改的是每次返回的分数，不应污染下一次验证/测试。
        self.assertEqual(model.scores[1, 1].item(), 100.)

    def test_test_loader_masks_training_only_as_required_by_original_protocol(self):
        train = pd.DataFrame({"user_id": [1], "movie_id": [1]})
        test = pd.DataFrame({"user_id": [1], "movie_id": [3]})
        loader = EvalDataLoader(test, train, batch_size=1, device="cpu")
        self.assertEqual(loader.get_user_train_pos_items([1]), [{1}])
        self.assertEqual(loader.get_user_eval_pos_items([1]), [{3}])
        # 电影2可代表验证交互；原协议在测试时仍保留它，因此测试答案排第2。
        result = EvaluationOnly(FixedScores(), "cpu")._evaluate(loader, topk=(1, 2))
        self.assertEqual(result["HR@1"], 0.)
        self.assertEqual(result["HR@2"], 1.)
        self.assertAlmostEqual(result["NDCG@2"], 1 / math.log2(3))


class SamplingProtocolTests(unittest.TestCase):
    def test_same_seed_produces_same_negatives_and_orders_after_different_models(self):
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [1, 2, 2, 3, 3, 4],
        })
        features = torch.ones(5, 7)
        features[0] = 0
        recorded = []
        for model_type in ("ncf", "hybrid"):
            seed_model(42)
            if model_type == "ncf":
                model = NCFRecommender(4, 5, [1, 2, 3, 4], embed_dim=2)
            else:
                model = HybridRecommender(4, 5, features, embed_dim=2)
            # 与正式runner相同：NumPy负采样和torch洗牌有独立的重置点。
            import numpy as np
            np.random.seed(42)
            loader = TrainDataLoader(train, batch_size=2, device="cpu")
            negatives = tensor_hash(loader.negative_items)
            for user, negative in zip(loader.users.tolist(), loader.negative_items.tolist()):
                self.assertNotIn(negative, loader.history_items[user])
            seed_shuffle(42)
            orders = []
            for _ in range(3):
                # 迭代batch不会训练模型，也不需要优化器。
                batches = list(loader)
                self.assertEqual(sum(batch.shape[1] for batch in batches), len(train))
                orders.append(tensor_hash(loader.users, loader.positive_items, loader.negative_items))
            recorded.append((negatives, orders))
            del model
        self.assertEqual(recorded[0], recorded[1])


def synthetic_run(model, seed, metric, negative_hash="same-negatives", orders=("epoch1", "epoch2")):
    return {
        "model": model,
        "seed": seed,
        "status": "completed",
        "validation": {"NDCG@10": metric},
        "test": {"NDCG@10": metric},
        "initial_negative_items_sha256": negative_hash,
        "history": [{"sample_order_sha256": order} for order in orders],
    }


class SummaryProtocolTests(unittest.TestCase):
    def test_aggregates_use_sample_std_and_distinguish_single_run(self):
        results = [
            synthetic_run("ncf", 42, 0.1),
            synthetic_run("ncf", 2026, 0.3),
            synthetic_run("hybrid", 42, 0.4),
            {"model": "popularity", "seed": None, "validation": {"NDCG@10": 0.05},
             "test": {"NDCG@10": 0.05}, "history": []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            save_summary(Path(directory), results)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            stats = {(row["model"], row["split"]): row for row in summary["aggregates"]}
            ncf = stats[("ncf", "test")]
            self.assertEqual(ncf["n_runs"], 2)
            self.assertAlmostEqual(ncf["mean"], 0.2)
            self.assertAlmostEqual(ncf["sample_std"], math.sqrt(0.02))
            self.assertIsNone(stats[("popularity", "test")]["sample_std"])
            self.assertIsNone(stats[("hybrid", "test")]["sample_std"])
            check, = summary["paired_sampling_checks"]
            self.assertEqual(check["seed"], 42)
            self.assertTrue(check["negative_samples_match"])
            self.assertTrue(check["epoch_sample_orders_match"])

    def test_mismatched_negatives_or_orders_fail_before_summary_is_written(self):
        base = [synthetic_run("ncf", 42, 0.1), synthetic_run("hybrid", 42, 0.2)]
        for mismatch in ("negative", "order"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                runs = copy.deepcopy(base)
                if mismatch == "negative":
                    runs[1]["initial_negative_items_sha256"] = "different"
                else:
                    runs[1]["history"][1]["sample_order_sha256"] = "different"
                with self.assertRaisesRegex(RuntimeError, "seed=42"):
                    save_summary(Path(directory), runs)
                self.assertFalse((Path(directory) / "summary.json").exists())
                self.assertFalse((Path(directory) / "summary.csv").exists())

    def test_early_stopping_checks_all_shared_epochs(self):
        results = [
            synthetic_run("ncf", 42, 0.1, orders=("epoch1", "epoch2", "epoch3")),
            synthetic_run("hybrid", 42, 0.2, orders=("epoch1", "epoch2")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            save_summary(Path(directory), results)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            check, = summary["paired_sampling_checks"]
            self.assertEqual(check["shared_epochs"], 2)
            self.assertTrue(check["epoch_sample_orders_match"])


if __name__ == "__main__":
    unittest.main()

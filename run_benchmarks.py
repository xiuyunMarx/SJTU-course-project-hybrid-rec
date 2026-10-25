"""在同一数据划分、负采样和训练预算下比较 Popularity、NCF 与 GPT-2 + NCF。"""

import os

# 必须在导入 torch 和创建 CUDA context 前设置。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import torch

from baselines import NCFRecommender, PopularityRecommender
from dataloader import EvalDataLoader, TrainDataLoader
from dataset import ML1MDataset
from recommender import HybridRecommender
from trainer import Trainer


def tensor_hash(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class RecordingTrainer(Trainer):
    """复用原训练/评测算法，只记录各轮损失、验证结果及样本顺序。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.best_epoch = None

    def _train_epoch(self):
        started = time.perf_counter()
        loss = super()._train_epoch()
        self.history.append({
            "epoch": len(self.history) + 1,
            "train_loss": float(loss),
            "train_seconds": time.perf_counter() - started,
            "sample_order_sha256": tensor_hash(
                self.train_loader.users,
                self.train_loader.positive_items,
                self.train_loader.negative_items,
            ),
        })
        return loss

    def _evaluate(self, eval_loader, topk=(10, 20)):
        result = super()._evaluate(eval_loader, topk=topk)
        if eval_loader is self.eval_loader:
            self.history[-1]["validation"] = result
            if result["NDCG@10"] > self.best_valid_score:
                self.best_epoch = self.history[-1]["epoch"]
        return result


class EvaluationOnly:
    """让没有可训练参数的热门推荐直接复用原评测器。"""

    _evaluate = Trainer._evaluate

    def __init__(self, model, device):
        self.model = model.to(device)


class TrainingSubset:
    """给 Popularity 提供与其他模型完全相同的训练子集（包括 --quick）。"""

    def __init__(self, dataset, train_data):
        self.items_df = dataset.items_df
        self.item_num = dataset.get_item_num()
        self.train_data = train_data

    def get_item_num(self):
        return self.item_num

    def get_split_data(self, split):
        if split != "train":
            raise ValueError("Popularity 只能读取训练数据")
        return self.train_data


class Tee:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def isatty(self):
        return False


@contextmanager
def capture_log(path):
    with path.open("x", encoding="utf-8") as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            yield


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def portable_path(path):
    """manifest 不记录用户名、主机地址或服务器绝对目录。"""
    path = Path(path)
    if path.is_absolute():
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return path.name
    return str(path)


def file_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        header = source.read(200)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path.name} 仍是 Git LFS 指针，请先执行 git lfs pull")
        digest.update(header)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": portable_path(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_shuffle(seed):
    # 模型大小不同会消耗不同数量的随机数，因此在初始化完成后单独重置。
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1)


def environment_info(device):
    versions = {}
    for name in ("numpy", "pandas", "torch", "tqdm", "transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "packages": versions,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_num_threads": torch.get_num_threads(),
    }


def preflight(args):
    """先验证参数与实际输入，再创建输出目录，失败时可直接修正后重试。"""
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("models 与 seeds 不允许包含重复值")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("seeds 必须在 [0, 2**32) 内")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.embed_dim) <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs、batch_size、eval_batch_size、embed_dim 和 lr 必须为正")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device 只支持 cpu、cuda 或 cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 无可用 CUDA；不会自动改用 CPU 训练")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("指定的 CUDA 设备编号不存在")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("output_dir 必须不存在或为空；请选择新的实验目录，避免覆盖已有结果")
    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise FileExistsError("checkpoint_dir 已存在且不是目录")
    for model in args.models:
        if model != "popularity":
            for seed in args.seeds:
                if (Path(args.checkpoint_dir) / f"{model}_seed{seed}.pth").exists():
                    raise FileExistsError("checkpoint_dir 中已有本次模型的权重；请选择新的 checkpoint_dir")

    fingerprints = {
        name: file_fingerprint(Path(args.data_path) / name)
        for name in ("users.dat", "movies.dat", "ratings.dat")
    }
    dataset = ML1MDataset(args.data_path)
    train_data, valid_data, test_data = dataset.get_split_data(None)
    if args.quick:
        selected = train_data["user_id"].drop_duplicates().head(500)
        train_data = train_data[train_data["user_id"].isin(selected)]
        valid_data = valid_data[valid_data["user_id"].isin(selected)]
        test_data = test_data[test_data["user_id"].isin(selected)]
    if any(part.empty for part in (train_data, valid_data, test_data)):
        raise ValueError("训练、验证、测试集合必须均非空")

    features = None
    if "hybrid" in args.models:
        fingerprints["text_features"] = file_fingerprint(args.features)
        features = torch.load(args.features, map_location="cpu", weights_only=True)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("文本特征必须是二维 Tensor")
        if features.shape[0] != dataset.get_item_num() or features.shape[1] == 0:
            raise ValueError("文本特征的行数必须等于电影 ID 空间大小，且列数必须为正")
        features = features.float()
        if not torch.isfinite(features).all():
            raise ValueError("文本特征含 NaN 或 Inf")
        catalog_mask = torch.zeros(dataset.get_item_num(), dtype=torch.bool)
        catalog_mask[torch.as_tensor(dataset.items_df.index.to_numpy(), dtype=torch.long)] = True
        feature_mask = features.abs().sum(dim=1).ne(0)
        if not torch.equal(feature_mask, catalog_mask):
            raise ValueError("文本非零行与电影目录不一致，无法保证三个模型使用相同候选集合")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device, dataset, (train_data, valid_data, test_data), features, fingerprints


def run_one(name, seed, args, device, dataset, splits, features):
    started = time.perf_counter()
    train_data, valid_data, test_data = splits
    valid_loader = EvalDataLoader(valid_data, train_data, args.eval_batch_size, device=device)
    test_loader = EvalDataLoader(test_data, train_data, args.eval_batch_size, device=device)
    result = {"model": name, "seed": seed, "status": "completed"}
    print(f"\nModel: {name}; seed: {seed}; device: {device}")
    if name == "popularity":
        model = PopularityRecommender(TrainingSubset(dataset, train_data))
        evaluator = EvaluationOnly(model, device)
        result.update({
            "trainable_parameters": 0,
            "validation": evaluator._evaluate(valid_loader),
            "test": evaluator._evaluate(test_loader),
            "history": [],
        })
    else:
        seed_model(seed)
        if name == "ncf":
            model = NCFRecommender(
                dataset.get_user_num(), dataset.get_item_num(), dataset.items_df.index, args.embed_dim
            )
        else:
            model = HybridRecommender(
                dataset.get_user_num(), dataset.get_item_num(), features, args.embed_dim
            )
        # 单独重置 NumPy：不同模型的参数初始化不会改变负样本。
        np.random.seed(seed)
        train_loader = TrainDataLoader(train_data, args.batch_size, shuffle=True, device=device)
        negative_hash = tensor_hash(train_loader.negative_items)
        trainer = RecordingTrainer(
            model, train_loader, valid_loader, test_loader,
            device=device, epochs=args.epochs, lr=args.lr,
            early_stop_patience=3,
        )
        seed_shuffle(seed)
        checkpoint = Path(args.checkpoint_dir) / f"{name}_seed{seed}.pth"
        print(f"Initial negative samples SHA256: {negative_hash}")
        valid_result, test_result = trainer.fit(save_model=True, model_path=str(checkpoint))
        result.update({
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "negative_sampling_seed": seed,
            "shuffle_seed": seed + 1,
            "initial_negative_items_sha256": negative_hash,
            "best_epoch": trainer.best_epoch,
            "epochs_completed": len(trainer.history),
            "validation": valid_result,
            "test": test_result,
            "history": trainer.history,
            "checkpoint": portable_path(checkpoint),
        })
    result["elapsed_seconds"] = time.perf_counter() - started
    print("Best validation:", result["validation"])
    print("Test:", result["test"])
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    return result


def save_summary(output_dir, results):
    aggregates = []
    for model in dict.fromkeys(result["model"] for result in results):
        runs = [result for result in results if result["model"] == model]
        for split in ("validation", "test"):
            for metric in runs[0][split]:
                values = [run[split][metric] for run in runs]
                aggregates.append({
                    "model": model, "split": split, "metric": metric, "n_runs": len(runs),
                    "mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                })
    checks = []
    for seed in dict.fromkeys(run["seed"] for run in results if run["seed"] is not None):
        paired = [run for run in results if run["seed"] == seed]
        if len(paired) < 2:
            continue
        negatives_match = len({run["initial_negative_items_sha256"] for run in paired}) == 1
        shared_epochs = min(len(run["history"]) for run in paired)
        orders_match = all(
            len({run["history"][i]["sample_order_sha256"] for run in paired}) == 1
            for i in range(shared_epochs)
        )
        checks.append({
            "seed": seed, "models": [run["model"] for run in paired],
            "negative_samples_match": negatives_match,
            "shared_epochs": shared_epochs, "epoch_sample_orders_match": orders_match,
        })
        if not negatives_match or not orders_match:
            raise RuntimeError(f"seed={seed} 的负采样或逐轮样本顺序不一致，请检查复现配置")
    write_json(output_dir / "summary.json", {
        "runs": results, "aggregates": aggregates, "paired_sampling_checks": checks,
        "std_note": "样本标准差使用 ddof=1；只有一次运行时为 null，不估计跨种子方差。",
    })
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["model", "split", "metric", "n_runs", "mean", "sample_std"])
        writer.writeheader()
        writer.writerows(aggregates)


def main(args):
    device, dataset, splits, features, fingerprints = preflight(args)
    config = vars(args).copy()
    for field in ("data_path", "features", "output_dir", "checkpoint_dir"):
        config[field] = portable_path(config[field])
    code_root = Path(__file__).resolve().parent
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": environment_info(device),
        "code": {
            name: file_fingerprint(code_root / name)
            for name in ("run_benchmarks.py", "baselines.py", "dataset.py", "dataloader.py", "recommender.py", "trainer.py", "gpt2.py", "requirements.txt")
        },
        "data_fingerprints": fingerprints,
        "dataset": {
            "split_random_state": 42, "split_ratio": [0.8, 0.1, 0.1],
            "item_id_space": dataset.get_item_num(), "catalog_items": len(dataset.items_df),
            "splits": {
                name: {"interactions": len(part), "users": int(part["user_id"].nunique())}
                for name, part in zip(("train", "validation", "test"), splits)
            },
        },
        "protocol": {
            "selection": "best validation NDCG@10; original Trainer.fit",
            "evaluation": "original Trainer._evaluate; full catalog; mask training history; topk=(10,20)",
            "negative_sampling": "one fixed training negative per interaction; NumPy seed equals model seed",
            "shuffle": "torch seed = model seed + 1, reset after model and loader initialization",
            "early_stop_patience": 3,
            "popularity": "training interaction counts; one deterministic evaluation",
            "quick": "first 500 distinct training users, same as main.py" if args.quick else None,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("output_dir 已被其他进程写入，请选择新的目录")
    if any(name != "popularity" for name in args.models):
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    results = []
    with capture_log(output_dir / "run.log"):
        write_json(output_dir / "manifest.json", manifest)
        print("Benchmark config:", json.dumps(config, ensure_ascii=False))
        print("Environment:", json.dumps(manifest["environment"], ensure_ascii=False))
        print("Dataset:", json.dumps(manifest["dataset"], ensure_ascii=False))
        for name in args.models:
            for seed in ([None] if name == "popularity" else args.seeds):
                stem = name if seed is None else f"{name}_seed{seed}"
                with capture_log(output_dir / f"{stem}.log"):
                    try:
                        result = run_one(name, seed, args, device, dataset, splits, features)
                    except Exception:
                        traceback.print_exc()
                        raise
                    write_json(output_dir / f"{stem}.json", result)
                results.append(result)
                save_summary(output_dir, results)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        print(f"Completed {len(results)} runs. Summary: {portable_path(output_dir / 'summary.csv')}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("popularity", "ncf", "hybrid"), default=["popularity", "ncf", "hybrid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--data_path", default="ml-1m")
    parser.add_argument("--features", default="movie_features.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/server1")
    parser.add_argument("--checkpoint_dir", default="artifacts/benchmarks")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

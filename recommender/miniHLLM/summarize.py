"""汇总 results/main 下的实验结果：均值 ± 样本标准差，整体与冷启动分桶。"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["popularity", "sasrec_id", "sasrec_id_400ep", "sasrec_text", "sasrec_id_text",
         "hllm_text", "hllm_text_frozen", "hllm_text_scratch", "hllm_text_scratch_lr1e-4", "hllm_text_id"]
METRICS = ["HR@10", "NDCG@10", "HR@20", "NDCG@20"]


def fmt(vals):
    vals = np.asarray(vals)
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"


def main(result_dir):
    runs = defaultdict(list)
    for path in sorted(Path(result_dir).glob("*_seed*.json")):
        tag = path.name.rsplit("_seed", 1)[0]
        runs[tag].append(json.loads(path.read_text()))
    tags = [t for t in ORDER if t in runs] + sorted(set(runs) - set(ORDER))

    print("## 测试集整体指标（均值 ± 样本标准差）\n")
    print("| 模型 | 种子数 | 最佳轮次 | " + " | ".join(METRICS) + " |")
    print("| --- | ---: | ---: | " + " | ".join("---:" for _ in METRICS) + " |")
    for tag in tags:
        rs = runs[tag]
        epochs = ", ".join(str(r["best_epoch"]) for r in rs)
        cells = [fmt([r["test"][m] for r in rs]) for m in METRICS]
        print(f"| {tag} | {len(rs)} | {epochs} | " + " | ".join(cells) + " |")

    buckets = list(runs[tags[0]][0]["test"]["buckets"].keys())
    print("\n## 按目标电影训练热度分桶的测试 NDCG@10\n")
    print("| 模型 | " + " | ".join(f"{b} (n={runs[tags[0]][0]['test']['buckets'][b]['n_users']})" for b in buckets) + " |")
    print("| --- | " + " | ".join("---:" for _ in buckets) + " |")
    for tag in tags:
        cells = [fmt([r["test"]["buckets"][b]["NDCG@10"] for r in runs[tag]]) for b in buckets]
        print(f"| {tag} | " + " | ".join(cells) + " |")

    print("\n## 训练耗时（秒，均值）\n")
    for tag in tags:
        print(f"- {tag}: {np.mean([r['train_seconds'] for r in runs[tag]]):.0f} s, "
              f"可训练参数 {runs[tag][0]['trainable_params']:,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "main")

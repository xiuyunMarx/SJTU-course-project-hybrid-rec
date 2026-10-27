"""从 results/main 生成报告用图：训练曲线、整体指标、冷启动分桶。"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "main"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# 固定顺序的分类色（已通过色觉验证），同一模型在所有图里用同一颜色。
COLORS = {
    "sasrec_id_400ep": "#2a78d6",
    "sasrec_id_text": "#eb6834",
    "hllm_text": "#1baf7a",
    "hllm_text_frozen": "#eda100",
    "hllm_text_scratch_lr1e-4": "#e87ba4",
    "sasrec_text": "#6250d6",
}
LABELS = {
    "popularity": "Popularity",
    "sasrec_id": "SASRec (ID, 200 ep)",
    "sasrec_id_400ep": "SASRec (ID)",
    "sasrec_text": "SASRec (text)",
    "sasrec_id_text": "SASRec (ID + text)",
    "hllm_text": "miniHLLM (text)",
    "hllm_text_frozen": "miniHLLM, frozen GPT-2",
    "hllm_text_scratch": "GPT-2 from scratch (lr 3e-4)",
    "hllm_text_scratch_lr1e-4": "GPT-2 from scratch",
    "hllm_text_id": "miniHLLM (text + ID)",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 200,
})


def load():
    runs = defaultdict(list)
    for path in sorted(RESULTS.glob("*_seed*.json")):
        runs[path.name.rsplit("_seed", 1)[0]].append(json.loads(path.read_text()))
    return runs


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_curves(runs):
    keys = ["sasrec_id_400ep", "sasrec_id_text", "hllm_text", "hllm_text_frozen", "hllm_text_scratch_lr1e-4"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k in keys:
        r = next(x for x in runs[k] if x["args"]["seed"] == 42)
        ep = [e["epoch"] for e in r["epochs"]]
        nd = [e["NDCG@10"] for e in r["epochs"]]
        ax.plot(ep, nd, color=COLORS[k], lw=2, label=LABELS[k])
        ax.plot(r["best_epoch"], max(nd), "o", color=COLORS[k], ms=5, mec="white", mew=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("validation NDCG@10")
    ax.set_title("Convergence (seed 42; dot = best epoch)", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "training_curves")


def fig_overall(runs):
    keys = ["popularity", "sasrec_id_400ep", "sasrec_text", "sasrec_id_text",
            "hllm_text_scratch_lr1e-4", "hllm_text_frozen", "hllm_text_id", "hllm_text"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, metric in zip(axes, ["HR@10", "NDCG@10"]):
        vals = [np.array([r["test"][metric] for r in runs[k]]) for k in keys]
        means = [v.mean() for v in vals]
        stds = [v.std(ddof=1) if len(v) > 1 else 0 for v in vals]
        y = np.arange(len(keys))
        ax.barh(y, means, xerr=stds, height=0.62, color="#2a78d6", ecolor=INK2, capsize=2, error_kw={"lw": 0.8})
        for yi, m in zip(y, means):
            ax.text(m + 0.006, yi, f"{m:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [LABELS[k] for k in keys])
        ax.set_xlim(0, max(means) * 1.25)
        ax.set_title(f"Test {metric}", loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.suptitle("Overall ranking quality (mean ± std over 3 seeds; Popularity is one run)",
                 x=0.01, y=1.04, ha="left", fontsize=9, color=INK2)
    save(fig, "overall_metrics")


def fig_buckets(runs):
    keys = ["sasrec_id_400ep", "sasrec_text", "sasrec_id_text", "hllm_text", "hllm_text_frozen"]
    buckets = list(runs[keys[0]][0]["test"]["buckets"].keys())
    names = {b: f"{b.replace('pop', 'train count ')}\n(n={runs[keys[0]][0]['test']['buckets'][b]['n_users']})" for b in buckets}
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    width = 0.15
    x = np.arange(len(buckets))
    for i, k in enumerate(keys):
        vals = np.array([[r["test"]["buckets"][b]["NDCG@10"] for r in runs[k]] for b in buckets])
        m, s = vals.mean(1), vals.std(1, ddof=1)
        pos = x + (i - (len(keys) - 1) / 2) * (width + 0.01)
        ax.bar(pos, m, width, yerr=s, color=COLORS[k], label=LABELS[k], ecolor=INK2, capsize=2, error_kw={"lw": 0.8})
        for p, v in zip(pos, m):
            ax.text(p, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5, color=INK, rotation=90, va="bottom")
    ax.set_xticks(x, [names[b] for b in buckets])
    ax.set_ylabel("test NDCG@10")
    ax.set_ylim(0, 0.29)
    ax.set_title("By target-item popularity in training data (mean ± std, 3 seeds)", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, fontsize=7, loc="upper left", columnspacing=0.8)
    save(fig, "coldstart_buckets")


if __name__ == "__main__":
    runs = load()
    fig_curves(runs)
    fig_overall(runs)
    fig_buckets(runs)
    print("wrote", sorted(p.name for p in FIG.iterdir()))

#!/usr/bin/env python3
"""Render the completed benchmark's real results; no synthetic data fallback.

Visual contract (course report, SVG + PNG):
  Source: summary.json from run_benchmarks.py, exactly seven completed runs.
  Questions: how do the two trainable models learn, and how do all three
  methods compare on the held-out test set under the fixed course protocol?
  Panels: (a) mean training BPR loss, (b) mean validation NDCG@10;
  separate figure: (a) test HR@10, (b) test NDCG@10.
  Uncertainty: three paired seeds, mean +/- sample standard deviation
  (ddof=1), not a confidence interval. Popularity has one deterministic run.
  Encoding: neutral gray / Okabe-Ito blue / vermilion, plus line style,
  markers and bar hatches; all metric and loss axes start at zero.
  Claims: show the measured comparison without asserting improvement,
  significance, convergence or transfer beyond this fixed data split.
  Traceability: figure metadata and plot_manifest.json record source SHA256.
  Placement: report's training-process and baseline-comparison discussion.

Run after benchmarks finish:
  python scripts/plot_benchmarks.py --results results/server1 --output_dir figures
Requires matplotlib (requirements-plot.txt pins version 3.9.4).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


MODELS = ("popularity", "ncf", "hybrid")
TRAINED_MODELS = ("ncf", "hybrid")
METRICS = ("HR@10", "NDCG@10")
LABELS = {"popularity": "Popularity", "ncf": "NCF", "hybrid": "NCF + GPT-2"}
COLORS = {"popularity": "#888888", "ncf": "#0072B2", "hybrid": "#D55E00"}
HATCHES = {"popularity": "", "ncf": "//", "hybrid": ".."}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label, unit_interval=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be a number")
    require(math.isfinite(value), f"{label} must be finite")
    require(0 <= value <= 1 if unit_interval else value >= 0,
            f"{label} is outside its valid range")
    return float(value)


def validate_summary(summary):
    """Reject partial, unpaired, inconsistent or non-finite experiment data."""
    runs = summary.get("runs", [])
    require(len(runs) == 7, "Expected seven completed runs: 1 Popularity + 3 NCF + 3 Hybrid")
    require(all(run.get("status") == "completed" for run in runs),
            "Every run must have status=completed")
    require({run.get("model") for run in runs} == set(MODELS), "Unexpected model set")
    by_model = {model: [run for run in runs if run["model"] == model] for model in MODELS}
    require(len(by_model["popularity"]) == 1, "Popularity must have one run")
    require(by_model["popularity"][0].get("seed") is None,
            "Popularity is deterministic and must not have a training seed")
    require(by_model["popularity"][0].get("history") == [],
            "Popularity must not have a training history")

    seeds = None
    epochs = None
    for model in TRAINED_MODELS:
        model_runs = by_model[model]
        require(len(model_runs) == 3, f"{model} requires three runs")
        model_seeds = [run.get("seed") for run in model_runs]
        require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in model_seeds),
                f"{model} has an invalid seed")
        require(len(set(model_seeds)) == 3, f"{model} requires three unique seeds")
        if seeds is None:
            seeds = set(model_seeds)
        require(set(model_seeds) == seeds, "NCF and Hybrid must use the same three seeds")
        by_model[model].sort(key=lambda run: run["seed"])
        for run in model_runs:
            history = run.get("history", [])
            require(bool(history), f"{model} seed {run['seed']} has no history")
            run_epochs = [entry.get("epoch") for entry in history]
            require(run_epochs == list(range(1, len(history) + 1)),
                    "Epoch records must be contiguous and start at one")
            require(run.get("epochs_completed") == len(history),
                    "epochs_completed does not match history")
            if epochs is None:
                epochs = run_epochs
            require(run_epochs == epochs, "All six training runs must have matching epochs")
            for entry in history:
                finite_number(entry.get("train_loss"), "Training loss")
                finite_number(entry.get("validation", {}).get("NDCG@10"),
                              "Validation NDCG@10", unit_interval=True)

    checks = summary.get("paired_sampling_checks", [])
    require(len(checks) == 3, "Expected three paired sampling checks")
    require({check.get("seed") for check in checks} == seeds,
            "Paired sampling checks must cover every training seed")
    for check in checks:
        require(set(check.get("models", [])) == set(TRAINED_MODELS),
                "Each paired check must compare NCF and Hybrid")
        require(check.get("negative_samples_match") is True
                and check.get("epoch_sample_orders_match") is True,
                f"Paired sampling check failed for seed {check.get('seed')}")
        require(check.get("shared_epochs") == len(epochs),
                "Paired sampling check does not cover every plotted epoch")

    aggregates = summary.get("aggregates", [])
    plotted = {}
    for model in MODELS:
        for metric in METRICS:
            matches = [row for row in aggregates if row.get("model") == model
                       and row.get("split") == "test" and row.get("metric") == metric]
            require(len(matches) == 1, f"Expected exactly one test aggregate for {model} {metric}")
            row = matches[0]
            values = [finite_number(run.get("test", {}).get(metric),
                                    f"{model} test {metric}", unit_interval=True)
                      for run in by_model[model]]
            require(row.get("n_runs") == len(values), "Aggregate run count mismatch")
            aggregate_mean = finite_number(row.get("mean"), "Aggregate mean", unit_interval=True)
            require(math.isclose(aggregate_mean, mean(values), rel_tol=1e-9, abs_tol=1e-12),
                    "Aggregate mean does not match run results")
            if len(values) == 1:
                require(row.get("sample_std") is None,
                        "Single-run Popularity must have null sample_std, not an estimated deviation")
            else:
                aggregate_std = finite_number(row.get("sample_std"), "Sample standard deviation")
                require(math.isclose(aggregate_std, stdev(values), rel_tol=1e-9, abs_tol=1e-12),
                        "Aggregate sample standard deviation does not match run results")
            plotted[model, metric] = row
    return by_model, epochs, sorted(seeds), plotted


def set_style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titlepad": 12,
        "axes.edgecolor": "#777777",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "hybrid-rec-benchmark-figures",
        "savefig.facecolor": "white",
    })


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.65)
    ax.tick_params(axis="both", length=3)


def save_figure(fig, output_dir, stem, source_hash, plt):
    description = (
        f"Source: benchmark summary.json; SHA256={source_hash}. "
        "NCF and Hybrid: mean +/- sample standard deviation over 3 seeds (ddof=1). "
        "Popularity: single deterministic run; no variance estimate."
    )
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.16,
                metadata={"Title": stem.replace("_", " "), "Description": description, "Date": None})
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0.16,
                metadata={"Description": description})
    plt.close(fig)
    return [svg_path, png_path]


def training_figure(by_model, epochs, seeds, output_dir, source_hash, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.22, wspace=0.27)
    fig.suptitle("Training on MovieLens-1M", x=0.5, y=0.98, fontsize=15, fontweight="bold")
    specifications = (
        ("(a) Training objective", "Mean BPR loss (lower is better)", lambda entry: entry["train_loss"]),
        ("(b) Validation ranking", "NDCG@10 (higher is better)", lambda entry: entry["validation"]["NDCG@10"]),
    )
    for ax, (title, ylabel, getter) in zip(axes, specifications):
        for model, marker, linestyle in (("ncf", "o", "-"), ("hybrid", "s", "--")):
            per_epoch = list(zip(*[[getter(entry) for entry in run["history"]]
                                  for run in by_model[model]]))
            means = [mean(values) for values in per_epoch]
            stds = [stdev(values) for values in per_epoch]
            ax.fill_between(epochs, [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=COLORS[model], alpha=0.17, linewidth=0)
            ax.plot(epochs, means, marker=marker, linestyle=linestyle,
                    color=COLORS[model], linewidth=2, markersize=5, label=LABELS[model])
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.set_xticks(epochs)
        ax.set_xlim(min(epochs) - 0.12, max(epochs) + 0.12)
        ax.set_ylim(bottom=0)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2)
    fig.text(0.5, 0.075, "Lines: mean; shaded bands: +/- sample std (ddof=1), 3 seeds: "
             + ", ".join(map(str, seeds)), ha="center", fontsize=9)
    fig.text(0.5, 0.025, "Fixed full-data split. Validation selects checkpoints; test data does not select epochs.",
             ha="center", fontsize=9, color="#555555")
    return save_figure(fig, output_dir, "training_curves", source_hash, plt)


def comparison_figure(plotted, seeds, output_dir, source_hash, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.26, wspace=0.26)
    fig.suptitle("Test-set baseline comparison", x=0.5, y=0.97, fontsize=15, fontweight="bold")
    for panel, (ax, metric) in enumerate(zip(axes, METRICS)):
        upper = max(plotted[model, metric]["mean"] + (plotted[model, metric]["sample_std"] or 0)
                    for model in MODELS)
        label_pad = max(upper * 0.035, 0.005)
        for index, model in enumerate(MODELS):
            row = plotted[model, metric]
            std = row["sample_std"]
            ax.bar(index, row["mean"], width=0.6, color=COLORS[model],
                   edgecolor="#333333", linewidth=0.6, hatch=HATCHES[model], zorder=3)
            if std is not None:
                ax.errorbar(index, row["mean"], yerr=std, fmt="none", ecolor="#222222",
                            elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
            ax.text(index, row["mean"] + (std or 0) + label_pad,
                    f"{row['mean']:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"({chr(97 + panel)}) {metric}")
        ax.set_ylabel(f"{metric} (higher is better)")
        ax.set_xticks(range(len(MODELS)), ["Popularity\nSingle run", "NCF\n3 seeds", "NCF + GPT-2\n3 seeds"])
        ax.set_ylim(0, max(upper * 1.22, 0.05))
        style_axis(ax)
    fig.text(0.5, 0.09, "NCF / NCF + GPT-2: mean +/- sample std (ddof=1), 3 seeds: "
             + ", ".join(map(str, seeds)), ha="center", fontsize=9)
    fig.text(0.5, 0.04, "Popularity: one deterministic evaluation; no variance estimate. Error bars are not confidence intervals.",
             ha="center", fontsize=9, color="#555555")
    return save_figure(fig, output_dir, "baseline_comparison", source_hash, plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/server1"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    source = args.results / "summary.json"
    payload = source.read_bytes()
    summary = json.loads(payload)
    by_model, epochs, seeds, plotted = validate_summary(summary)
    # The runner always writes a manifest before experiments start. Require it
    # so the figure's full-data label is backed by the recorded configuration.
    manifest_path = args.results / "manifest.json"
    require(manifest_path.is_file(), "The runner's manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("config", {}).get("quick") is False,
            "These course-report figures require the full-data run, not --quick")
    require(manifest.get("config", {}).get("epochs") == len(epochs),
            "History does not cover the configured training budget")
    require(sorted(manifest.get("config", {}).get("seeds", [])) == seeds,
            "The manifest and summary must use the same three seeds")

    # Import only after validation, allowing a clear error for incomplete inputs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_style(plt)
    source_hash = hashlib.sha256(payload).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = training_figure(by_model, epochs, seeds, args.output_dir, source_hash, plt)
    files += comparison_figure(plotted, seeds, args.output_dir, source_hash, plt)
    traceability = {
        "source": "summary.json",
        "source_sha256": source_hash,
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib": matplotlib.__version__,
        "completed_runs": 7,
        "seeds": seeds,
        "epochs": epochs,
        "uncertainty": "sample standard deviation, ddof=1; not confidence intervals",
        "popularity": "single deterministic run; sample_std=null; no error bar",
        "paired_sampling_checks": "passed for all three seeds and every plotted epoch",
        "outputs": [path.name for path in files],
    }
    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(traceability, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for path in files:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

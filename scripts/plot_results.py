"""Export measured results as standard scientific plots for the repository."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results")
    parser.add_argument("--calibration", default="results/calibration_lightgbm_final.json")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding="utf-8")) for p in Path(args.runs).rglob("metrics.json")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = ["lr", "lightgbm", "deepfm", "sharedbottom", "mmoe", "deepfm_matched"]
    labels = {"lr": "LR", "lightgbm": "LightGBM", "deepfm": "DeepFM", "sharedbottom": "Shared-Bottom",
              "mmoe": "MMoE", "deepfm_matched": "DeepFM (matched)"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for split, offset, color, title in [("standard_aligned_valid", -.10, "#3068bc", "Standard exposures"),
                                        ("random_valid", .10, "#e28035", "Random exposures")]:
        values, errors = [], []
        for model in order:
            metrics = [r["metrics"][split][r["tasks"][0]]["auc"] for r in records if r["model"] == model]
            values.append(np.mean(metrics))
            errors.append(np.std(metrics, ddof=1) if len(metrics) > 1 else 0)
        ax.errorbar(values, np.arange(len(order)) + offset, xerr=errors, fmt="o", capsize=3,
                    color=color, label=title)
    ax.set_yticks(np.arange(len(order)), [labels[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("Primary-task ROC AUC (long_view)")
    ax.set_title("Date-aligned policy evaluation: Apr 26-30, 2022")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower left")
    fig.text(.02, .01, "Fixed 250k training exposures. Neural models: 3 seeds. Error bars are seed SD, not confidence intervals.", fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, 1])
    fig.savefig(out / "policy_auc.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for model in order[2:]:
        selected = [r for r in records if r["model"] == model]
        epochs = sorted({v["epoch"] for r in selected for v in r["curves"]})
        means = [np.mean([v["valid_primary_logloss"] for r in selected for v in r["curves"] if v["epoch"] == epoch]) for epoch in epochs]
        ax.plot(epochs, means, marker="o", label=labels[model])
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation primary LogLoss (lower is better)")
    ax.set_title("Early-stopping diagnostics on real exposure logs")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=180)
    plt.close(fig)
    calibration = Path(args.calibration)
    if calibration.exists():
        data = json.loads(calibration.read_text(encoding="utf-8"))
        splits = list(data["metrics"])
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        states = ["raw", "constant_calibration_prior", "calibrated"]
        for ax, metric in zip(axes, ["logloss", "brier"]):
            for i, split in enumerate(splits):
                values = [data["metrics"][split][state][metric] for state in states]
                ax.bar(np.arange(3) + (i - (len(splits) - 1) / 2) * .34, values, width=.32,
                       label=split.replace("_", " "))
            ax.set_xticks(np.arange(3), ["Raw model", "Calibration prior", "Platt scaling"])
            ax.set_ylabel(metric + " (lower is better)")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Random-exposure probability calibration: LightGBM")
        fig.tight_layout()
        fig.savefig(out / "calibration.png", dpi=180)
        plt.close(fig)
    print(f"Saved experiment figures to {out}")

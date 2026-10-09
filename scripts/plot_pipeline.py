"""Plot the actual end-to-end benchmark independently of exposure metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.loads(Path("results/end_to_end.json").read_text(encoding="utf-8"))
out = Path("results/plots")
out.mkdir(parents=True, exist_ok=True)
names = ["popular", "itemcf", "dual_tower", "fusion"]
labels = ["Popular", "ItemCF", "Dual tower", "RRF fusion"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
x = np.arange(4)
for offset, split, color in [(-.18, "validation", "#346aca"), (.18, "test", "#de8533")]:
    axes[0].bar(x + offset, [data[split]["retrieval"][n]["recall@300"] for n in names], .34, label=split, color=color)
axes[0].set_xticks(x, labels)
axes[0].set_ylabel("Observed-positive Recall@300")
axes[0].set_title("Candidate retrieval")
axes[0].legend()
blends = data["validation"]["ranking_blends"]
weights = sorted(blends, key=float)
axes[1].plot([float(w) for w in weights], [blends[w]["without_author_cap"]["ndcg@10"] for w in weights], marker="o", label="Before author cap")
axes[1].plot([float(w) for w in weights], [blends[w]["with_author_cap"]["ndcg@10"] for w in weights], marker="o", label="After author cap")
axes[1].axvline(data["ranking_model_weight"], color="#777", linestyle="--", label="Validation selection")
axes[1].set_xlabel("Ranking-model weight (0 = retrieval, 1 = model)")
axes[1].set_ylabel("Validation observed-positive NDCG@10")
axes[1].set_title("Ranking blend selection before test")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
fig.suptitle("KuaiRank: real closed-catalog recommendation pipeline")
fig.text(.02, .012, f"{data['catalog_videos']:,} training-known videos; 1,000 eligible users/day; observed positives only, not online utility.", fontsize=8)
fig.tight_layout(rect=[0, .035, 1, 1])
fig.savefig(out / "end_to_end.png", dpi=180)
print("Saved", out / "end_to_end.png")

"""Diagnose short-list inflation and report nontrivial logged-exposure ranking."""
import argparse
import json
from pathlib import Path

import numpy as np

from kuai_rank.data import save_json
from kuai_rank.metrics import group_ranges


def diagnostics(path, min_candidates=5, k=10):
    data = np.load(path, allow_pickle=False)
    y, p = data["targets"][:, 0], data["probabilities"][:, 0]
    users, days = data["users"].astype(str), data["days"].astype(str)
    keys = np.char.add(np.char.add(users, ":"), days)
    order, boundaries = group_ranges(keys)
    sizes, discounts, values = np.diff(boundaries), 1 / np.log2(np.arange(k) + 2), []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        idx = order[start:end]
        positives = int(y[idx].sum())
        if len(idx) < min_candidates or not 0 < positives < len(idx):
            continue
        top = idx[np.argsort(-p[idx], kind="stable")[:k]]
        dcg = float((y[top] * discounts[:len(top)]).sum())
        idcg = float(discounts[:min(k, positives)].sum())
        values.append(dcg / idcg)
    return {"logged_user_day_group_count": len(sizes),
            "group_size_quantiles": dict(zip(["p50", "p90", "p95", "p99", "max"], map(float, np.quantile(sizes, [.5, .9, .95, .99, 1])))),
            "single_candidate_group_fraction": float((sizes == 1).mean()),
            "eligible_groups": len(values), "min_candidates": min_candidates,
            "requires_positive_and_negative": True,
            "nontrivial_logged_ndcg@10": float(np.mean(values)) if values else None,
            "scope": "logged exposure sets; not full-catalog recommendation quality"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results")
    parser.add_argument("--out", default="results/ranking_diagnostics.json")
    args = parser.parse_args()
    results = []
    for path in sorted(Path(args.runs).rglob("metrics.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        for split, info in run["predictions"].items():
            results.append({"model": run["model"], "seed": run["seed"], "split": split,
                            **diagnostics(path.parent / info["file"])})
    save_json(args.out, {"rows": results})
    print(f"Saved {len(results)} ranking diagnostics to {args.out}")

"""Paired user-cluster bootstrap for primary-task LogLoss differences.

Compares exactly matched prediction events. Reports uncertainty across users
for one fixed trained-model pair, not training-seed uncertainty or causal lift.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kuai_rank.data import save_json


def paired_bootstrap(baseline, candidate, repetitions=2000, seed=2026):
    a, b = np.load(baseline, allow_pickle=False), np.load(candidate, allow_pickle=False)
    for key in ["event_id", "targets", "users", "days"]:
        if not np.array_equal(a[key][:, :1] if key == "targets" else a[key],
                              b[key][:, :1] if key == "targets" else b[key]):
            raise ValueError(f"Predictions are not paired on {key}")
    y = a["targets"][:, 0]
    pa, pb = np.clip(a["probabilities"][:, 0], 1e-7, 1 - 1e-7), np.clip(b["probabilities"][:, 0], 1e-7, 1 - 1e-7)
    loss_a = -(y * np.log(pa) + (1 - y) * np.log1p(-pa))
    loss_b = -(y * np.log(pb) + (1 - y) * np.log1p(-pb))
    frame = pd.DataFrame({"user": a["users"], "delta": loss_b - loss_a})
    clustered = frame.groupby("user").agg(delta=("delta", "sum"), count=("delta", "size"))
    values, counts = clustered.delta.to_numpy(), clustered["count"].to_numpy()
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(clustered), len(clustered))
        differences.append(float(values[indices].sum() / counts[indices].sum()))
    low, high = np.quantile(differences, [.025, .975])
    return {"metric": "primary_logloss_candidate_minus_baseline", "delta": float((loss_b - loss_a).mean()),
            "ci95": [float(low), float(high)], "users": len(clustered), "rows": len(y),
            "resamples": repetitions, "seed": seed, "lower_is_better": True,
            "scope": "paired user-cluster uncertainty for a fixed model pair; not causal lift or training uncertainty"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = paired_bootstrap(args.baseline, args.candidate)
    save_json(args.out, result)
    print(result)

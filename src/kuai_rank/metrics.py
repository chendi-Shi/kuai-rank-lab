"""Exposure classification and logged user-day ranking; no causal lift claims."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(y, p, bins=15):
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    if len(y) == 0:
        return {"rows": 0, "auc": None, "logloss": None, "brier": None, "ece": None}
    indices = np.minimum((p * bins).astype(int), bins - 1)
    calibration, ece = [], 0.0
    for i in range(bins):
        mask = indices == i
        if mask.any():
            confidence, observed = float(p[mask].mean()), float(y[mask].mean())
            count = int(mask.sum())
            ece += count / len(y) * abs(confidence - observed)
            calibration.append({"bin": i, "rows": count, "predicted": confidence, "observed": observed})
    return {"rows": len(y), "positive_rate": float(y.mean()), "mean_prediction": float(p.mean()),
            "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
            "logloss": float(log_loss(y, p, labels=[0, 1])), "brier": float(np.mean((p - y) ** 2)),
            "ece": float(ece), "calibration": calibration}


def group_ranges(groups):
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]), len(order)]
    return order, boundaries


def grouped_metrics(y, p, users, days, k=10):
    y, p = np.asarray(y), np.asarray(p)
    # Mann-Whitney AUC with average score ranks handles ties exactly, while
    # avoiding tens of thousands of separate sklearn calls on a full split.
    grouped = pd.DataFrame({"user": np.asarray(users).astype(str), "y": y, "p": p})
    grouped["positive_rank"] = grouped.groupby("user")["p"].rank(method="average") * grouped.y
    stats = grouped.groupby("user").agg(n=("y", "size"), positives=("y", "sum"), rank_sum=("positive_rank", "sum"))
    stats = stats[(stats.positives > 0) & (stats.positives < stats.n)]
    auc = ((stats.rank_sum - stats.positives * (stats.positives + 1) / 2) /
           (stats.positives * (stats.n - stats.positives)))
    weights, aucs, eligible_users = int(stats.n.sum()), float((stats.n * auc).sum()), len(stats)
    keys = np.char.add(np.char.add(np.asarray(users).astype(str), ":"), np.asarray(days).astype(str))
    order, boundaries = group_ranges(keys)
    ndcg, positive_groups = [], 0
    discount = 1 / np.log2(np.arange(k) + 2)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        positives = int(y[idx].sum())
        if positives == 0:
            continue
        positive_groups += 1
        # Ties use stable event order, consistently across models.
        top = idx[np.argsort(-p[idx], kind="stable")[:k]]
        dcg = float((y[top] * discount[:len(top)]).sum())
        idcg = float(discount[:min(k, positives)].sum())
        ndcg.append(dcg / idcg)
    return {"user_gauc": aucs / weights if weights else None,
            "gauc_eligible_users": eligible_users, "gauc_row_coverage": weights / len(y) if len(y) else 0,
            f"logged_user_day_ndcg@{k}": float(np.mean(ndcg)) if ndcg else None,
            "ndcg_positive_groups": positive_groups,
            "ndcg_all_groups": len(boundaries) - 1,
            "ranking_protocol": "observed user-day exposure sets; groups without positives excluded"}


def evaluate(frame, probabilities, tasks):
    result = {}
    for i, task in enumerate(tasks):
        y, p = frame[task].to_numpy(), probabilities[:, i]
        result[task] = binary_metrics(y, p)
        # Grouped metrics focus on the primary task to keep full-data audits efficient.
        if i == 0:
            result[task].update(grouped_metrics(y, p, frame.user_id, frame.day))
            masks = {"user_no_prior_exposures": frame.user_count_log.eq(0).to_numpy(),
                     "user_with_history": frame.user_count_log.gt(0).to_numpy(),
                     "item_no_prior_exposures": frame.item_count_log.eq(0).to_numpy(),
                     "item_low_exposure": frame.item_count_log.lt(np.log1p(10)).to_numpy(),
                     "item_high_exposure": frame.item_count_log.ge(np.log1p(100)).to_numpy()}
            result[task]["slices"] = {name: binary_metrics(y[mask], p[mask]) for name, mask in masks.items()}
    return result

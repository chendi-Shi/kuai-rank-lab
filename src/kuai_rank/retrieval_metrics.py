"""Observed-positive retrieval benchmark, not fully judged catalog utility."""
import numpy as np

from .data import DAY_MS


def targets_for_day(frame, snapshot, max_users=1000, seed=2026):
    day = frame.loc[frame.time_ms.ge(snapshot.meta["cutoff_ms"]) &
                    frame.time_ms.lt(snapshot.meta["cutoff_ms"] + DAY_MS) & frame.long_view.eq(1)]
    catalog = {v: i for i, v in enumerate(snapshot.video_ids)}
    targets, repeated, outside, raw = {}, 0, 0, 0
    for user, group in day.groupby("user_id"):
        items = set(group.video_id.astype(str))
        seen = set(snapshot.seen_indices(user))
        keep = set()
        for video in items:
            raw += 1
            if video not in catalog:
                outside += 1
            elif catalog[video] in seen:
                repeated += 1
            else:
                keep.add(catalog[video])
        if keep:
            targets[str(user)] = keep
    keys = np.asarray(sorted(targets))
    if len(keys) > max_users:
        keys = np.sort(np.random.default_rng(seed).choice(keys, max_users, replace=False))
    return {str(u): targets[str(u)] for u in keys}, {
        "raw_positive_user_item_pairs": raw, "previously_seen_pairs_excluded": repeated,
        "outside_catalog_pairs": outside, "eligible_positive_pairs": raw - outside - repeated,
        "eligible_users": len(targets), "evaluated_users": len(keys),
        "catalog_positive_coverage": (raw - outside) / raw if raw else None,
        "scope": "First-day observed long-view positives, within training-known catalog, excluding prior standard exposures."}


def ranking_metrics(indices, targets, k):
    top = np.asarray(indices)[:k]
    hits = np.asarray([int(i in targets) for i in top], dtype=np.float64)
    discounts = 1 / np.log2(np.arange(len(top)) + 2)
    ideal = (1 / np.log2(np.arange(min(len(targets), k)) + 2)).sum()
    return {f"recall@{k}": float(hits.sum() / len(targets)),
            f"hit@{k}": float(hits.any()), f"ndcg@{k}": float((hits * discounts).sum() / ideal)}


def mean_metrics(rows):
    if not rows:
        raise ValueError("No eligible evaluation users")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def exact_validation(model, snapshot, targets, k=100):
    import torch
    model.eval()
    known = {u: i + 1 for i, u in enumerate(snapshot.user_ids)}
    users = list(targets)
    with torch.inference_mode():
        items = model.item_vectors(torch.arange(len(snapshot.video_ids))).numpy()
        queries = model.user_vectors(torch.tensor([known.get(u, 0) for u in users])).numpy()
    scores = queries @ items.T
    results = []
    for row, user in enumerate(users):
        scores[row, snapshot.seen_indices(user)] = -np.inf
        ids = np.arange(len(items))
        order = np.lexsort((ids, -scores[row]))[:k]
        results.append(ranking_metrics(order, targets[user], k))
    return mean_metrics(results)

"""Point-in-time features and consumed-item profiles for candidate generation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .data import TASKS, DAY_MS, OFFSET_MS, day_boundary, file_sha256, save_json
from .integrity import check_files


def interaction_matrix(frame, users, videos, positives_only=False):
    selected = frame[frame.long_view.eq(1)] if positives_only else frame
    pairs = selected[["user_id", "video_id"]].astype(str).drop_duplicates()
    u = pd.Index(users).get_indexer(pairs.user_id)
    i = pd.Index(videos).get_indexer(pairs.video_id)
    keep = (u >= 0) & (i >= 0)
    return sparse.csr_matrix((np.ones(keep.sum(), dtype=np.float32), (u[keep], i[keep])),
                             shape=(len(users), len(videos)))


def summarize(frame, entity, prefix):
    stats = frame.groupby(entity).agg(count=("long_view", "size"), **{task: (task, "sum") for task in TASKS})
    out = pd.DataFrame(index=stats.index.astype(str))
    out[f"{prefix}_count_log"] = np.log1p(stats["count"].to_numpy()).astype(np.float32)
    for task, prior in zip(TASKS, [.3, .4, .02]):
        out[f"{prefix}_{task}_rate"] = ((stats[task].to_numpy() + 10 * prior) / (stats["count"].to_numpy() + 10)).astype(np.float32)
    return out


def history_before(data, cutoff):
    # The deployed snapshots are at validation/test start. Test labels never
    # enter either snapshot, even though hashing their files is permitted.
    parts = [pd.read_parquet(Path(data) / "train.parquet")]
    if cutoff > parts[0].time_ms.max():
        parts.append(pd.read_parquet(Path(data) / "valid.parquet"))
    frame = pd.concat(parts, ignore_index=True)
    return frame.loc[frame.time_ms.lt(cutoff) & frame.traffic.eq("standard")].copy()


def build_snapshot(history, metadata, cutoff, root, catalog_ids=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if (cutoff + OFFSET_MS) % DAY_MS:
        raise ValueError("Snapshot cutoff must be midnight UTC+8")
    history = history.loc[history.time_ms.lt(cutoff) & history.traffic.eq("standard")].copy()
    if history.empty:
        raise ValueError("No history preceding snapshot")
    for key in ["user_id", "video_id", "author_id"]:
        history[key] = history[key].astype(str)
    ids = sorted(history.video_id.unique()) if catalog_ids is None else list(catalog_ids)
    if not set(ids) <= set(history.video_id):
        raise ValueError("Catalog contains videos not observed before snapshot")
    meta = metadata.copy()
    meta.video_id = meta.video_id.astype(str)
    meta.author_id = meta.author_id.astype(str)
    if meta.video_id.duplicated().any():
        raise ValueError("Duplicate catalog metadata")
    catalog = meta.set_index("video_id").loc[ids].copy()
    uploads = pd.to_datetime(catalog.upload_dt, errors="coerce")
    date = pd.Timestamp(cutoff, unit="ms", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None)
    if (uploads > date.normalize()).any():
        raise ValueError("Catalog upload date lies after snapshot")
    # Use duration already observed in ordinary pre-cutoff logs. Do not infer
    # duration from the future evaluation event or from future aggregates.
    catalog["duration_log"] = history.groupby("video_id").duration_log.median().reindex(ids).to_numpy(dtype=np.float32)
    width = pd.to_numeric(catalog.server_width, errors="coerce").fillna(0)
    height = pd.to_numeric(catalog.server_height, errors="coerce").fillna(0)
    catalog["aspect_ratio"] = (width / height.clip(lower=1)).clip(0, 5).astype(np.float32)
    catalog["video_age_log"] = np.log1p((date.normalize() - uploads).dt.days.fillna(0).clip(lower=0)).astype(np.float32)
    catalog["tag_first"] = catalog.tag.fillna("UNKNOWN").astype(str).str.split(",").str[0]
    for prefix, entity in [("item", "video_id"), ("author", "author_id")]:
        stats = summarize(history, entity, prefix)
        keys = catalog.index if entity == "video_id" else catalog.author_id
        for name in stats:
            catalog[name] = stats[name].reindex(keys).to_numpy()
    users = sorted(history.user_id.unique())
    user_stats = summarize(history, "user_id", "user").reindex(users)
    catalog.reset_index().to_parquet(root / "catalog.parquet", index=False)
    user_stats.rename_axis("user_id").reset_index().to_parquet(root / "users.parquet", index=False)
    sparse.save_npz(root / "seen.npz", interaction_matrix(history, users, ids))
    sparse.save_npz(root / "positives.npz", interaction_matrix(history, users, ids, True))
    popularity = history.loc[history.long_view.eq(1)].groupby("video_id").size().reindex(ids, fill_value=0).to_numpy()
    np.save(root / "popularity.npy", popularity)
    manifest = {"cutoff_ms": cutoff, "history_rows": len(history), "max_history_ms": int(history.time_ms.max()),
                "catalog_videos": len(ids), "users": len(users), "history_policy": "standard only, strictly before cutoff",
                "duration_source": "median observed duration_log strictly before cutoff",
                "files": {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}}
    save_json(root / "snapshot.json", manifest)
    return manifest


class FeatureSnapshot:
    def __init__(self, root):
        root = Path(root)
        self.meta = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
        check_files(root, self.meta["files"])
        self.catalog = pd.read_parquet(root / "catalog.parquet").set_index("video_id")
        self.users = pd.read_parquet(root / "users.parquet").set_index("user_id")
        self.video_ids = list(self.catalog.index)
        self.user_ids = list(self.users.index)
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.seen = sparse.load_npz(root / "seen.npz")
        self.positives = sparse.load_npz(root / "positives.npz")
        self.popularity = np.load(root / "popularity.npy", allow_pickle=False)

    def seen_indices(self, user):
        u = self.user_index.get(str(user))
        return self.seen[u].indices if u is not None else np.array([], dtype=np.int64)

    def positive_indices(self, user):
        u = self.user_index.get(str(user))
        return self.positives[u].indices if u is not None else np.array([], dtype=np.int64)

    def features(self, user, indices, timestamp_ms=None, tab="0"):
        stamp = self.meta["cutoff_ms"] + 12 * 3600000 if timestamp_ms is None else timestamp_ms
        if not self.meta["cutoff_ms"] <= stamp < self.meta["cutoff_ms"] + DAY_MS:
            raise ValueError("Request date differs from feature snapshot date; rebuild the snapshot")
        frame = self.catalog.iloc[indices].reset_index().copy()
        frame["user_id"], frame["tab"] = str(user), str(tab)
        row = self.users.loc[str(user)] if str(user) in self.users.index else None
        frame["user_count_log"] = float(row.user_count_log) if row is not None else 0.
        for task, prior in zip(TASKS, [.3, .4, .02]):
            frame[f"user_{task}_rate"] = float(row[f"user_{task}_rate"]) if row is not None else prior
        date = pd.Timestamp(stamp, unit="ms", tz="UTC").tz_convert("Asia/Shanghai")
        hour = date.hour + date.minute / 60
        frame["hour_sin"], frame["hour_cos"] = np.float32(np.sin(hour * 2 * np.pi / 24)), np.float32(np.cos(hour * 2 * np.pi / 24))
        frame["weekday_sin"], frame["weekday_cos"] = np.float32(np.sin(date.dayofweek * 2 * np.pi / 7)), np.float32(np.cos(date.dayofweek * 2 * np.pi / 7))
        return frame

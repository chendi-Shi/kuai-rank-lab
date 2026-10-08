"""Chronological splits and conservative histories available before the request day.

Only standard exposures update history. Random traffic is scored using the same
standard-traffic history, and never silently becomes ordinary training data.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["long_view", "is_click", "is_like"]
LOG_COLUMNS = ["user_id", "video_id", "time_ms", "tab", "duration_ms", *TASKS]
BASIC_COLUMNS = ["video_id", "author_id", "video_type", "upload_type", "music_type",
                 "tag", "upload_dt", "server_width", "server_height"]
DAY_MS = 86400000
OFFSET_MS = 8 * 3600000


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def day_boundary(date):
    return int(pd.Timestamp(date, tz="Asia/Shanghai").timestamp() * 1000)


def validate_config(config):
    if config["tasks"] != TASKS:
        raise ValueError("This version's target contract is long_view/is_click/is_like in that order")
    allowed_cats = {"user_id", "video_id", "author_id", "tab", "video_type", "upload_type", "music_type", "tag_first"}
    allowed_nums = {"duration_log", "aspect_ratio", "video_age_log", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"}
    for prefix in ["user", "item", "author"]:
        allowed_nums.add(f"{prefix}_count_log")
        allowed_nums.update(f"{prefix}_{task}_rate" for task in TASKS)
    if not set(config["categorical"]) <= allowed_cats or not set(config["numeric"]) <= allowed_nums:
        raise ValueError("Feature contract includes unsupported or outcome-leaking columns")
    if len(set(config["categorical"])) != len(config["categorical"]) or len(set(config["numeric"])) != len(config["numeric"]):
        raise ValueError("Duplicate feature names")
    if not config["categorical"] or not config["numeric"]:
        raise ValueError("At least one categorical and one numeric feature are required")
    bounds = [day_boundary(config[key]) for key in ["train_end", "random_calibration_end", "random_valid_end"]]
    if not bounds[0] <= bounds[1] < bounds[2] or config["valid_end"] != config["random_valid_end"]:
        raise ValueError("Validation dates must support policy-aligned evaluation")
    for name in ["epochs", "batch_size", "hidden_dim", "embedding_dim", "experts", "threads"]:
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if len(config["task_weights"]) != len(TASKS) or min(config["task_weights"]) < 0 or config["task_weights"][0] <= 0:
        raise ValueError("Task weights must be nonnegative with a positive primary weight")


def validate_logs(frame):
    required = set(LOG_COLUMNS) | {"traffic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing log columns: {sorted(missing)}")
    if frame[LOG_COLUMNS].isna().any().any():
        raise ValueError("Null log values; inspect raw files before training")
    for col in TASKS:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"Nonbinary target: {col}")
    if not frame.traffic.isin(["standard", "random"]).all():
        raise ValueError("Unknown traffic policy")
    if (frame.duration_ms < 0).any():
        raise ValueError("Negative video duration")
    return {"rows": len(frame), "zero_duration": int((frame.duration_ms == 0).sum()),
            "duplicate_log_rows": int(frame.duplicated(LOG_COLUMNS + ["traffic"]).sum()),
            "min_time_ms": int(frame.time_ms.min()), "max_time_ms": int(frame.time_ms.max()),
            "users": int(frame.user_id.nunique()), "videos": int(frame.video_id.nunique())}


def split_labels(frame, config):
    t = frame.time_ms
    standard = frame.traffic.eq("standard")
    result = np.full(len(frame), "random_test", dtype=object)
    result[standard & (t < day_boundary(config["train_end"]))] = "train"
    result[standard & (t >= day_boundary(config["train_end"])) &
           (t < day_boundary(config["valid_end"]))] = "valid"
    result[standard & (t >= day_boundary(config["valid_end"]))] = "test"
    result[~standard & (t < day_boundary(config["random_calibration_end"]))] = "random_calibration"
    result[~standard & (t >= day_boundary(config["random_calibration_end"])) &
           (t < day_boundary(config["random_valid_end"]))] = "random_valid"
    return result


def daily_history(frame, entity, prefix):
    """All events on day D see the cumulative standard history through D-1.

    Timestamp ties and delayed intra-day feedback cannot leak across examples.
    The previous-day feedback is assumed finalized; this is a daily replay,
    rather than a claim of instantaneous production feedback availability.
    """
    standard = frame.loc[frame.traffic.eq("standard"), [entity, "day", *TASKS]]
    daily = standard.groupby([entity, "day"], sort=True).agg(
        count=("long_view", "size"), **{task: (task, "sum") for task in TASKS}).reset_index()
    values = ["count", *TASKS]
    daily[values] = daily.groupby(entity, sort=False)[values].cumsum()
    left = frame[[entity, "day"]].copy()
    left["_row"] = np.arange(len(left))
    matched = pd.merge_asof(left.sort_values("day", kind="stable"),
                            daily.sort_values("day", kind="stable"),
                            on="day", by=entity, allow_exact_matches=False, direction="backward")
    matched = matched.sort_values("_row")[values].fillna(0)
    count = matched["count"].to_numpy(dtype=np.float64)
    result = {f"{prefix}_count_log": np.log1p(count).astype(np.float32)}
    # Fixed priors, not estimates using validation/test labels.
    for task, prior in zip(TASKS, [0.3, 0.4, 0.02]):
        positives = matched[task].to_numpy(dtype=np.float64)
        result[f"{prefix}_{task}_rate"] = ((positives + 10 * prior) / (count + 10)).astype(np.float32)
    return result


def build_features(frame, metadata):
    basic = metadata[BASIC_COLUMNS].copy()
    if basic.video_id.duplicated().any():
        raise ValueError("Duplicate video metadata keys")
    result = frame.merge(basic, on="video_id", how="left", validate="many_to_one", sort=False)
    result["author_id"] = result.author_id.fillna(-1).astype(np.int64)
    result["day"] = ((result.time_ms + OFFSET_MS) // DAY_MS).astype(np.int64)
    date = pd.to_datetime(result.time_ms, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    hour = date.dt.hour + date.dt.minute / 60
    result["hour_sin"] = np.sin(hour * 2 * np.pi / 24).astype(np.float32)
    result["hour_cos"] = np.cos(hour * 2 * np.pi / 24).astype(np.float32)
    result["weekday_sin"] = np.sin(date.dt.dayofweek * 2 * np.pi / 7).astype(np.float32)
    result["weekday_cos"] = np.cos(date.dt.dayofweek * 2 * np.pi / 7).astype(np.float32)
    result["duration_log"] = np.log1p(result.duration_ms / 1000).astype(np.float32)
    width = pd.to_numeric(result.server_width, errors="coerce").fillna(0)
    height = pd.to_numeric(result.server_height, errors="coerce").fillna(0)
    result["aspect_ratio"] = (width / height.clip(lower=1)).clip(0, 5).astype(np.float32)
    upload = pd.to_datetime(result.upload_dt, errors="coerce", utc=True)
    # Upload metadata is date-only. Age is intentionally rounded to days.
    age = ((date.dt.tz_localize(None).dt.normalize() - upload.dt.tz_localize(None)).dt.total_seconds() / 86400)
    result["video_age_log"] = np.log1p(age.fillna(0).clip(lower=0)).astype(np.float32)
    result["tag_first"] = result.tag.fillna("UNKNOWN").astype(str).str.split(",").str[0]
    for entity, prefix in [("user_id", "user"), ("video_id", "item"), ("author_id", "author")]:
        print(f"Building prior-day {prefix} history", flush=True)
        for name, value in daily_history(result, entity, prefix).items():
            result[name] = value
    return result


def prepare(raw, out, config):
    validate_config(config)
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for pattern, traffic in [("log_standard*.csv", "standard"), ("log_random*.csv", "random")]:
        paths = sorted(raw.rglob(pattern))
        if not paths:
            raise FileNotFoundError(f"Missing {pattern} under {raw}")
        for path in paths:
            print(f"Reading {path.name}", flush=True)
            df = pd.read_csv(path, usecols=LOG_COLUMNS)
            df["traffic"] = traffic
            frames.append(df)
            sources.append({"file": path.name, "sha256": file_sha256(path), "rows": len(df)})
    frame = pd.concat(frames, ignore_index=True)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    audit = validate_logs(frame)
    basic_paths = sorted(raw.rglob("video_features_basic*.csv"))
    if len(basic_paths) != 1:
        raise ValueError("Expected exactly one Pure basic-video feature file")
    metadata = pd.read_csv(basic_paths[0], usecols=BASIC_COLUMNS)
    sources.append({"file": basic_paths[0].name, "sha256": file_sha256(basic_paths[0])})
    missing_meta = ~frame.video_id.isin(metadata.video_id)
    audit["missing_video_metadata_rows"] = int(missing_meta.sum())
    frame["split"] = split_labels(frame, config)
    features = build_features(frame, metadata)
    selected = ["event_id", "user_id", "video_id", "time_ms", "day", "traffic", "split", *TASKS,
                *config["categorical"], *config["numeric"]]
    selected = list(dict.fromkeys(selected))
    features = features[selected].sort_values(["time_ms", "event_id"], kind="stable")
    for name in config["categorical"]:
        features[name] = features[name].fillna("UNKNOWN").astype(str)
    if not np.isfinite(features[config["numeric"]].to_numpy()).all():
        raise ValueError("Nonfinite model features")
    split_audit = {}
    expected = ["train", "valid", "test", "random_calibration", "random_valid", "random_test"]
    for split in expected:
        part = features.loc[features.split.eq(split)]
        if part.empty:
            raise ValueError(f"Empty required split: {split}")
        part.to_parquet(out / f"{split}.parquet", index=False)
        split_audit[split] = {"rows": len(part), "users": int(part.user_id.nunique()),
                              "videos": int(part.video_id.nunique()),
                              "min_time_ms": int(part.time_ms.min()), "max_time_ms": int(part.time_ms.max()),
                              "positive_rates": {task: float(part[task].mean()) for task in TASKS}}
    if not split_audit["train"]["max_time_ms"] < split_audit["valid"]["min_time_ms"]:
        raise ValueError("Train/validation temporal overlap")
    if not split_audit["valid"]["max_time_ms"] < split_audit["test"]["min_time_ms"]:
        raise ValueError("Validation/test temporal overlap")
    audit["splits"] = split_audit
    save_json(out / "audit.json", audit)
    contract = {"config": config, "sources": sources, "history": "prior UTC+8 days; standard exposures only",
                "user_snapshot_features": "excluded: snapshot time not established",
                "monthly_video_statistics": "excluded: month-wide future information",
                "ranking_protocol": "logged exposure sets grouped by user and UTC+8 day, not full-catalog ranking",
                "feedback_assumption": "previous-day outcomes finalized; observed-policy rolling daily replay",
                "pure_limit": "history is observed candidate-pool history, not the full user sequence"}
    save_json(out / "manifest.json", contract)
    print(json.dumps(audit, indent=2), flush=True)

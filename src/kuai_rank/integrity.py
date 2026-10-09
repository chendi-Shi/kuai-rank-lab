"""Verify bytes before reading trusted experiment artifacts or datasets."""
from pathlib import Path

from .data import file_sha256

SPLITS = ("train", "valid", "test", "random_calibration", "random_valid", "random_test")


def check_files(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or Path(name).name != name:
            raise ValueError(f"Artifact name must be a local filename: {name}")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Artifact missing or changed since training: {name}")


def dataset_snapshot(root):
    return {f"{split}.parquet": file_sha256(Path(root) / f"{split}.parquet") for split in SPLITS}


def verify_dataset(root, meta):
    if file_sha256(Path(root) / "manifest.json") != meta["data_manifest_sha256"]:
        raise ValueError("Prepared dataset manifest changed since training")
    hashes = meta.get("prepared_split_sha256")
    if hashes is not None:
        if set(hashes) != {f"{split}.parquet" for split in SPLITS}:
            raise ValueError("Incomplete prepared-split fingerprints")
        check_files(root, hashes)
    # Old reports retain their original bytes; do not fabricate missing hashes.
    return {"manifest_verified": True, "prepared_split_bytes_verified": hashes is not None}


def verify_models(root, meta):
    kind = meta["model"]
    model_file = "linear.pkl" if kind == "lr" else "lightgbm.txt" if kind == "lightgbm" else "checkpoint.pt"
    hashes = meta.get("model_artifact_sha256", {})
    if not {"encoder.pkl", model_file} <= set(hashes):
        raise ValueError("Required model artifact fingerprints are missing")
    check_files(root, hashes)


def verify_predictions(root, meta):
    records = meta.get("predictions", {})
    if not records:
        raise ValueError("Prediction fingerprints are missing")
    check_files(root, {entry["file"]: entry["sha256"] for entry in records.values()})
    return len(records)

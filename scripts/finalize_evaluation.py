"""Freeze validation-selected models and evaluate existing checkpoints once.

No refitting, early stopping or model selection uses final test labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kuai_rank.data import file_sha256, save_json
from kuai_rank.inference import Scorer
from kuai_rank.metrics import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/initial")
    parser.add_argument("--data", default="data/processed")
    args = parser.parse_args()
    runs, data = Path(args.runs), Path(args.data)
    paths = sorted(runs.rglob("metrics.json"))
    if not paths:
        raise ValueError("No completed validation runs")
    freeze_path = runs / "test_freeze.json"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    model_losses = {}
    for model in sorted({r["model"] for r in records}):
        values = [r["metrics"]["valid"][r["tasks"][0]]["logloss"] for r in records if r["model"] == model]
        model_losses[model] = float(np.mean(values))
    freeze = {"utc_time": datetime.now(timezone.utc).isoformat(),
              "selection_metric": "mean validation primary LogLoss",
              "selected_model": min(model_losses, key=model_losses.get), "validation_losses": model_losses,
              "runs_before_test": {str(path): file_sha256(path) for path in paths},
              "data_manifest_sha256": file_sha256(data / "manifest.json")}
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if set(freeze["runs_before_test"]) != {str(path) for path in paths}:
            raise ValueError("Run set changed since test freeze")
    else:
        if any(record["test_evaluated"] for record in records):
            raise ValueError("A run already accessed test before this freeze")
        save_json(freeze_path, freeze)
    for path in paths:
        scorer = Scorer(path.parent)
        meta = scorer.meta
        if meta["test_evaluated"]:
            if meta.get("test_freeze_sha256") != file_sha256(freeze_path):
                raise ValueError("Completed test belongs to a different freeze")
            print(f"Existing completed frozen test: {path}", flush=True)
            continue
        if file_sha256(path) != freeze["runs_before_test"][str(path)]:
            raise ValueError("Validation run changed after test freeze")
        if meta["data_manifest_sha256"] != freeze["data_manifest_sha256"]:
            raise ValueError("Prepared dataset changed after training")
        start = time.perf_counter()
        for split in ["test", "random_test"]:
            print(f"Evaluating frozen {meta['model']} seed {meta['seed']}: {split}", flush=True)
            part = pd.read_parquet(data / f"{split}.parquet")
            p = scorer.predict(part)
            meta["metrics"][split] = evaluate(part, p, meta["tasks"])
            meta["unknown_category_rates"][split] = scorer.encoder.unknown_rates(part)
            destination = path.parent / f"predictions_{split}.npz"
            np.savez_compressed(destination, event_id=part.event_id.to_numpy(), probabilities=p,
                                targets=part[meta["tasks"]].to_numpy(), users=part.user_id.to_numpy(dtype=str), days=part.day.to_numpy())
            meta["predictions"][split] = {"file": destination.name, "sha256": file_sha256(destination)}
        meta["test_evaluated"] = True
        meta["test_evaluation_seconds"] = time.perf_counter() - start
        meta["test_freeze_sha256"] = file_sha256(freeze_path)
        save_json(path, meta)
    from kuai_rank.report import compare_runs
    compare_runs(runs, runs / "comparison.json")

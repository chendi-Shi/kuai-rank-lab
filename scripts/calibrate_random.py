"""Fit primary-task Platt scaling only on the reserved random calibration split.

This is supervised domain calibration, not exposure propensity estimation or
causal debiasing. It cannot establish online CTR/engagement improvements.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kuai_rank.data import file_sha256, save_json
from kuai_rank.inference import Scorer
from kuai_rank.metrics import binary_metrics
from kuai_rank.integrity import verify_dataset


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))[:, None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("Calibration report already exists; use a new destination")
    data, run = Path(args.data), Path(args.run)
    scorer = Scorer(run)
    verify_dataset(data, scorer.meta)
    primary = scorer.meta["tasks"][0]
    calibration = pd.read_parquet(data / "random_calibration.parquet")
    p = scorer.predict(calibration)[:, 0]
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=2026)
    model.fit(logit(p), calibration[primary].to_numpy())
    calibration_prior = float(calibration[primary].mean())
    parameters = {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
    report = {"base_model": scorer.meta["model"], "seed": scorer.meta["seed"], "primary_task": primary,
              "parameters": parameters, "calibration_rows": len(calibration), "calibration_prior": calibration_prior,
              "calibration_sha256": file_sha256(data / "random_calibration.parquet"),
              "base_artifacts": scorer.meta["model_artifact_sha256"], "metrics": {},
              "scope": "Platt scaling from random calibration labels; supervised probability calibration, not causal debiasing"}
    splits = ["random_valid"]
    if args.evaluate_test:
        if not scorer.meta["test_evaluated"] or not scorer.meta.get("test_freeze_sha256"):
            raise ValueError("Freeze and finalize the base-model experiment before opening final test")
        splits.append("random_test")
    # Parameters are fixed before any validation/test outcomes are evaluated.
    for split in splits:
        frame = pd.read_parquet(data / f"{split}.parquet")
        scores = scorer.predict(frame)[:, 0]
        calibrated = model.predict_proba(logit(scores))[:, 1]
        report["metrics"][split] = {"raw": binary_metrics(frame[primary], scores),
                                    "calibrated": binary_metrics(frame[primary], calibrated),
                                    "constant_calibration_prior": binary_metrics(frame[primary], np.full(len(frame), calibration_prior))}
    save_json(args.out, report)
    print(json.dumps({"parameters": parameters, "metrics": {split: {
        state: {key: metrics[key] for key in ["auc", "logloss", "brier", "ece"]}
        for state, metrics in values.items()} for split, values in report["metrics"].items()}}, indent=2))

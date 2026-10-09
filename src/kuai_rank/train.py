import copy
import json
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import torch
from torch import nn

from .data import save_json, file_sha256, day_boundary, validate_config
from .encoding import FeatureEncoder
from .metrics import binary_metrics, evaluate
from .models import RankModel, capacity_matched_hidden, parameter_budget
from .integrity import dataset_snapshot, check_files


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predict_neural(model, cats, nums, batch_size=8192):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(cats), batch_size):
            c = torch.from_numpy(cats[start:start + batch_size])
            n = torch.from_numpy(nums[start:start + batch_size])
            predictions.append(model(c, n).sigmoid().numpy())
    return np.concatenate(predictions)


def fit_neural(cats, nums, y, vcats, vnums, vy, sizes, kind, config):
    model = RankModel(sizes, nums.shape[1], kind=kind,
                      embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"],
                      experts=config["experts"], n_tasks=len(config["tasks"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    weights = torch.tensor(config["task_weights"][:model.n_tasks], dtype=torch.float32)
    weights = weights / weights.sum()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    curves = []
    for epoch in range(config["epochs"]):
        start = time.perf_counter()
        model.train()
        indices = np.random.permutation(len(y))
        total = 0.0
        for offset in range(0, len(indices), config["batch_size"]):
            idx = indices[offset:offset + config["batch_size"]]
            logits = model(torch.from_numpy(cats[idx]), torch.from_numpy(nums[idx]))
            labels = torch.from_numpy(y[idx, :model.n_tasks])
            loss = (loss_fn(logits, labels) * weights).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        p = predict_neural(model, vcats, vnums)
        validation = binary_metrics(vy[:, 0], p[:, 0])
        record = {"epoch": epoch + 1, "loss": total / len(y),
                  "valid_primary_logloss": validation["logloss"], "valid_primary_auc": validation["auc"],
                  "seconds": time.perf_counter() - start}
        curves.append(record)
        print(json.dumps(record), flush=True)
        if validation["logloss"] < best_loss:
            best_loss, best_state, bad_epochs = validation["logloss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return model, curves


def run(data_dir, result_dir, kind, config, seed, evaluate_test=False):
    validate_config(config)
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, config["threads"])
    prepared_hashes = dataset_snapshot(data_dir)
    started = time.perf_counter()
    full_train = pd.read_parquet(data_dir / "train.parquet")
    # Same training rows for every model and every seed: the seed controls model randomness only.
    cap = config["max_train_rows"]
    train = full_train.sample(n=min(cap, len(full_train)), random_state=2026).sort_values("event_id") if cap else full_train
    valid = pd.read_parquet(data_dir / "valid.parquet")
    encoder = FeatureEncoder(config["categorical"], config["numeric"]).fit(train)
    cats, nums = encoder.transform(train)
    vcats, vnums = encoder.transform(valid)
    capacity_target = None
    effective_kind = kind
    if kind == "deepfm_matched":
        capacity_target = parameter_budget(encoder.sizes, nums.shape[1], "mmoe", config["embedding_dim"],
                                           config["hidden_dim"], config["experts"], len(config["tasks"]))
        config = copy.deepcopy(config)
        config["hidden_dim"] = capacity_matched_hidden(encoder.sizes, nums.shape[1], config["embedding_dim"],
                                                     config["hidden_dim"], config["experts"], len(config["tasks"]))
        effective_kind = "deepfm"
    tasks = config["tasks"] if kind in {"sharedbottom", "mmoe"} else config["tasks"][:1]
    y = train[config["tasks"]].to_numpy(dtype=np.float32)
    vy = valid[config["tasks"]].to_numpy(dtype=np.float32)
    curves = []
    if kind == "lightgbm":
        x, vx = np.column_stack([cats, nums]), np.column_stack([vcats, vnums])
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=100, reg_lambda=1.0,
                                   random_state=seed, n_jobs=config["threads"], verbosity=-1,
                                   deterministic=True, force_col_wise=True)
        model.fit(x, y[:, 0], eval_set=[(vx, vy[:, 0])], eval_metric="binary_logloss",
                  categorical_feature=list(range(cats.shape[1])),
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        predict = lambda c, n: model.predict_proba(np.column_stack([c, n]))[:, 1:2]
        model.booster_.save_model(str(result_dir / "lightgbm.txt"))
        feature_names = config["categorical"] + config["numeric"]
        save_json(result_dir / "feature_importance.json",
                  dict(zip(feature_names, model.booster_.feature_importance(importance_type="gain").tolist())))
        parameters = None
    elif kind == "lr":
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        x = hstack([onehot.fit_transform(cats), csr_matrix(nums)], format="csr")
        model = LogisticRegression(C=0.1, max_iter=300, solver="liblinear", random_state=seed)
        model.fit(x, y[:, 0])
        predict = lambda c, n: model.predict_proba(hstack([onehot.transform(c), csr_matrix(n)], format="csr"))[:, 1:2]
        parameters = int(model.coef_.size + model.intercept_.size)
        with (result_dir / "linear.pkl").open("wb") as f:
            pickle.dump((onehot, model), f)
    else:
        model, curves = fit_neural(cats, nums, y, vcats, vnums, vy, encoder.sizes, effective_kind, config)
        predict = lambda c, n: predict_neural(model, c, n)
        parameters = sum(p.numel() for p in model.parameters())
        torch.save(model.state_dict(), result_dir / "checkpoint.pt")
        if kind == "mmoe":
            with torch.inference_mode():
                gates = model.gate_mean(torch.from_numpy(vcats[:8192]), torch.from_numpy(vnums[:8192]))
            save_json(result_dir / "gate_mean.json", {task: row for task, row in zip(tasks, gates.tolist())})
    with (result_dir / "encoder.pkl").open("wb") as f:
        pickle.dump(encoder, f)
    evaluations, prediction_files, unknowns = {}, {}, {}
    # Final test is explicitly opt-in after the experiment configuration is frozen.
    splits = ["valid", "standard_aligned_valid", "random_valid"]
    if evaluate_test:
        splits += ["test", "random_test"]
    for split in splits:
        print(f"Evaluating {split}", flush=True)
        if split == "standard_aligned_valid":
            part = valid[(valid.time_ms >= day_boundary(config["random_calibration_end"])) &
                         (valid.time_ms < day_boundary(config["random_valid_end"]))]
            if part.empty:
                raise ValueError("No standard validation exposures in the random-validation window")
        else:
            part = valid if split == "valid" else pd.read_parquet(data_dir / f"{split}.parquet")
        c, n = encoder.transform(part)
        p = predict(c, n)
        evaluations[split] = evaluate(part, p, tasks)
        unknowns[split] = encoder.unknown_rates(part)
        prediction_path = result_dir / f"predictions_{split}.npz"
        np.savez_compressed(prediction_path, event_id=part.event_id.to_numpy(), probabilities=p,
                            targets=part[tasks].to_numpy(), users=part.user_id.to_numpy(dtype=str), days=part.day.to_numpy())
        prediction_files[split] = {"file": prediction_path.name, "sha256": file_sha256(prediction_path)}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        revision = ""
    check_files(data_dir, prepared_hashes)
    source_files = ["data.py", "encoding.py", "metrics.py", "models.py", "train.py", "integrity.py"]
    code_fingerprints = {name: file_sha256(Path(__file__).parent / name) for name in source_files}
    summary = {"model": kind, "effective_model_kind": effective_kind,
               "matched_mmoe_parameter_target": capacity_target,
               "seed": seed, "tasks": tasks, "config": config,
               "source_sha256": code_fingerprints,
               "model_artifact_sha256": {path.name: file_sha256(path) for path in result_dir.iterdir()
                                         if path.suffix in {".pkl", ".pt", ".txt"}},
               "training_rows": len(train), "available_training_rows": len(full_train),
               "training_event_ids_sha256": __import__("hashlib").sha256(train.event_id.to_numpy().tobytes()).hexdigest(),
               "parameters": parameters, "seconds": time.perf_counter() - started,
               "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
               "prepared_split_sha256": prepared_hashes,
               "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                               "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__,
                               "threads": config["threads"], "device": "cpu", "git_revision": revision},
               "curves": curves, "unknown_category_rates": unknowns,
               "predictions": prediction_files, "metrics": evaluations,
               "test_evaluated": evaluate_test}
    save_json(result_dir / "metrics.json", summary)
    print(json.dumps({"model": kind, "seed": seed, "train_rows": len(train), "seconds": summary["seconds"],
                      "valid_auc": evaluations["valid"][tasks[0]]["auc"]}), flush=True)
    return summary

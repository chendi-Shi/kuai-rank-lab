"""Validation-only LR ablation with the original fixed training-event sample."""
import argparse
import copy
import json
from pathlib import Path

from kuai_rank.data import save_json
from kuai_rank.train import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/initial/lr_s2026")
    parser.add_argument("--out", default="artifacts/history_ablation/lr_no_history_s2026")
    parser.add_argument("--report", default="results/history_ablation.json")
    args = parser.parse_args()
    baseline = json.loads((Path(args.baseline) / "metrics.json").read_text(encoding="utf-8"))
    config = copy.deepcopy(baseline["config"])
    removed = [name for name in config["numeric"] if name.startswith(("user_", "item_", "author_"))]
    config["numeric"] = [name for name in config["numeric"] if name not in removed]
    config["threads"] = 1
    result = run("data/processed", args.out, "lr", config, baseline["seed"], evaluate_test=False)
    if result["training_event_ids_sha256"] != baseline["training_event_ids_sha256"]:
        raise ValueError("Ablation training sample differs from baseline")
    if result["data_manifest_sha256"] != baseline["data_manifest_sha256"]:
        raise ValueError("Ablation prepared dataset differs from baseline")
    primary = baseline["tasks"][0]
    rows = {split: {"with_history": baseline["metrics"][split][primary],
                    "without_history": result["metrics"][split][primary]}
            for split in ["valid", "standard_aligned_valid", "random_valid"]}
    save_json(args.report, {"hypothesis": "Prior-day behavioral histories explain much of the strong LR baseline.",
                           "removed_features": removed, "test_accessed": False,
                           "scope": "Validation-only feature ablation after the original test was reported; no new test selection.",
                           "ablation_run": result, "metrics": rows})
    print(json.dumps({split: {kind: {k: v[k] for k in ["auc", "logloss", "user_gauc"]}
                             for kind, v in values.items()} for split, values in rows.items()}, indent=2))

"""Compare MMoE against both single-task controls across training seeds."""
import json
from pathlib import Path

from paired_bootstrap import paired_bootstrap
from kuai_rank.data import save_json


if __name__ == "__main__":
    runs = {}
    for path in Path("results").rglob("metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        runs[record["model"], record["seed"]] = path.parent
    rows = []
    for seed in [2026, 2027, 2028]:
        for baseline in ["deepfm", "deepfm_matched"]:
            result = paired_bootstrap(runs[baseline, seed] / "predictions_valid.npz",
                                      runs["mmoe", seed] / "predictions_valid.npz")
            rows.append({"baseline": baseline, "candidate": "mmoe", "training_seed": seed, **result})
            print(seed, baseline, result["delta"], result["ci95"], flush=True)
    save_json("results/multiseed_paired_bootstrap.json", {"split": "valid", "rows": rows})

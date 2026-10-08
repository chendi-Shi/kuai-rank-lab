"""Sequential CPU experiments with isolated, non-overwritten result directories."""
import argparse
import json
from pathlib import Path
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pure.json")
    parser.add_argument("--out", default="results/initial")
    parser.add_argument("--models", nargs="+", default=["lr", "lightgbm", "deepfm", "sharedbottom", "mmoe"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from kuai_rank.train import run
    seeds = args.seeds or config["seeds"]
    for model in args.models:
        model_seeds = seeds[:1] if model in {"lr", "lightgbm"} else seeds
        for seed in model_seeds:
            destination = Path(args.out) / f"{model}_s{seed}"
            if (destination / "metrics.json").exists():
                print(f"Existing completed run: {destination}", flush=True)
                continue
            if destination.exists():
                raise RuntimeError(f"Incomplete run at {destination}; inspect it before rerunning")
            print(f"Training {model}, seed={seed}, threads={config['threads']} -> {destination}", flush=True)
            run("data/processed", destination, model, config, seed, args.evaluate_test)
    from kuai_rank.report import compare_runs
    compare_runs(args.out, Path(args.out) / "comparison.json")

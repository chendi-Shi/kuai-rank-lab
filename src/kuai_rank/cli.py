import argparse
import json
from pathlib import Path

from .data import prepare


def main():
    parser = argparse.ArgumentParser(description="KuaiRank: real-exposure multi-task ranking experiments")
    parser.add_argument("--config", default="configs/pure.json")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--raw", default="data/raw/KuaiRand-Pure/data")
    data.add_argument("--out", default="data/processed")
    train = commands.add_parser("train")
    train.add_argument("--model", required=True, choices=["lr", "lightgbm", "deepfm", "deepfm_matched", "sharedbottom", "mmoe"])
    train.add_argument("--data", default="data/processed")
    train.add_argument("--out", required=True)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--evaluate-test", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--runs", default="results/initial")
    compare.add_argument("--out", default="results/comparison.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "prepare":
        prepare(args.raw, args.out, config)
    elif args.command == "train":
        from .train import run
        run(args.data, args.out, args.model, config, args.seed, args.evaluate_test)
    else:
        from .report import compare_runs
        compare_runs(args.runs, args.out)


if __name__ == "__main__":
    main()

"""Create an API example from prepared real exposure features, without labels."""
import argparse
import json
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/valid.parquet")
    parser.add_argument("--config", default="configs/pure.json")
    parser.add_argument("--out", default="artifacts/request_example.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.data)
    counts = frame.groupby(["user_id", "day"]).size()
    user, day = counts.idxmax()
    part = frame[(frame.user_id == user) & (frame.day == day)].head(100)
    candidates = [{"candidate_id": str(row.event_id),
                   "categorical": {col: str(row[col]) for col in config["categorical"]},
                   "numeric": {col: float(row[col]) for col in config["numeric"]}} for _, row in part.iterrows()]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidates": candidates, "k": 10}, indent=2), encoding="utf-8")
    print(f"Saved {len(candidates)} real-exposure candidate feature rows to {path}")

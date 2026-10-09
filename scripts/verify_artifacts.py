"""Read-only verification; does not load pickle files or evaluate test labels."""
import argparse
import json
from pathlib import Path

from kuai_rank.data import file_sha256, save_json
from kuai_rank.integrity import verify_dataset, verify_models, verify_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="results/artifact_verification.json")
    args = parser.parse_args()
    paths = sorted(Path(args.runs).rglob("metrics.json"))
    if not paths:
        raise ValueError("No run reports found")
    rows = []
    for path in paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        coverage = verify_dataset(args.data, meta)
        verify_models(path.parent, meta)
        count = verify_predictions(path.parent, meta)
        rows.append({"model": meta["model"], "seed": meta["seed"], "metrics_sha256": file_sha256(path),
                     "model_bytes_verified": True, "prediction_files_verified": count, **coverage})
    save_json(args.out, {"runs": len(rows), "rows": rows,
                        "scope": "Local byte verification. Legacy runs lack split-byte hashes; this is explicitly reported."})
    print(f"Verified {len(rows)} runs and {sum(r['prediction_files_verified'] for r in rows)} prediction files")

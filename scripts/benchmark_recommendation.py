"""HTTP benchmark for recall -> features -> rank -> author diversification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

from kuai_rank.data import save_json


def request(url, payload):
    start = time.perf_counter()
    req = urllib.request.Request(url + "/recommend", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    return (time.perf_counter() - start) * 1000, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="results/recommendation_benchmark.json")
    parser.add_argument("--example", default="results/end_to_end_example.json")
    parser.add_argument("--snapshot", default="artifacts/end_to_end/test_snapshot")
    args = parser.parse_args()
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    example = json.loads(Path(args.example).read_text())
    user = example["user_id"]
    _, online_example = request(args.url, {"user_id": user})
    assert online_example["ranking"] == example["ranking"], "HTTP output differs from saved offline pipeline example"
    with urllib.request.urlopen(args.url + "/", timeout=30) as response:
        assert "从用户 ID 到推荐列表" in response.read().decode("utf-8")
    for _ in range(10):
        request(args.url, {"user_id": user})
    # Include several actual users plus a missing-ID cold-start request.
    import pandas as pd
    users = pd.read_parquet(Path(args.snapshot) / "users.parquet").user_id.astype(str).tolist()[:20]
    users.append("cold-user-not-in-training")
    payloads = [{"user_id": users[i % len(users)]} for i in range(args.requests)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda body: request(args.url, body), payloads))
    elapsed = time.perf_counter() - started
    client = np.asarray([r[0] for r in results])
    server = np.asarray([r[1]["timing_ms"]["total"] for r in results])
    for _, response in results:
        ids = [r["video_id"] for r in response["ranking"]]
        assert len(ids) == len(set(ids))
        counts = {}
        for row in response["ranking"]:
            counts[row["author_id"]] = counts.get(row["author_id"], 0) + 1
        assert max(counts.values(), default=0) <= 2
    _, cold = request(args.url, {"user_id": "cold-user-not-in-training"})
    try:
        request(args.url, {"user_id": user, "timestamp_ms": 0})
        raise AssertionError("Stale snapshot request unexpectedly succeeded")
    except urllib.error.HTTPError as error:
        assert error.code == 422
    report = {"requests": args.requests, "concurrency": args.concurrency, "distinct_users": len(users),
              "qps": args.requests / elapsed, "http_ms": dict(zip(["p50", "p95", "p99"], map(float, np.quantile(client, [.5, .95, .99])))),
              "pipeline_ms": dict(zip(["p50", "p95", "p99"], map(float, np.quantile(server, [.5, .95, .99])))),
              "cold_start_ok": cold["cold_user"] and bool(cold["ranking"]), "duplicate_and_author_caps_ok": True,
              "offline_http_ranking_identical": True, "demo_page_available": True,
              "stale_snapshot_rejected": True, "scope": "Local CPU historical-snapshot full pipeline; no external database/network feature store."}
    save_json(args.out, report)
    save_json(Path(args.out).parent / "recommendation_http_example.json", results[0][1])
    print(json.dumps(report, indent=2))

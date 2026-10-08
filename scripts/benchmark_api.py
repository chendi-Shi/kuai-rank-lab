"""Local HTTP benchmark for the ranker, including serialization and round trip."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import sys
import time
import urllib.request

import numpy as np


def request_once(url, body):
    start = time.perf_counter()
    request = urllib.request.Request(url + "/rank", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())
    return (time.perf_counter() - start) * 1000, float(result["ranking_ms"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--payload", default="artifacts/request_example.json")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="results/serving_benchmark.json")
    args = parser.parse_args()
    # Explicit empty proxy handler keeps localhost requests on the local machine.
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    body = Path(args.payload).read_bytes()
    with urllib.request.urlopen(args.url + "/health", timeout=30) as response:
        health = json.loads(response.read())
    for _ in range(20):
        request_once(args.url, body)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        timings = list(pool.map(lambda _: request_once(args.url, body), range(args.requests)))
    elapsed = time.perf_counter() - start
    client = np.array([t[0] for t in timings])
    scoring = np.array([t[1] for t in timings])
    report = {"model": health, "requests": args.requests, "concurrency": args.concurrency,
              "candidates": len(json.loads(body)["candidates"]), "warmup_requests": 20,
              "qps": args.requests / elapsed, "client_ms": dict(zip(["p50", "p95", "p99"], map(float, np.quantile(client, [.5, .95, .99])))),
              "scoring_ms": dict(zip(["p50", "p95", "p99"], map(float, np.quantile(scoring, [.5, .95, .99])))),
              "environment": {"python": sys.version, "platform": platform.platform()},
              "scope": "local warm-model HTTP ranker test; repeated fixed real-exposure feature payload; no retrieval or feature-store latency"}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

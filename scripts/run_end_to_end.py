"""Build, select, freeze and evaluate a real closed-catalog recommendation pipeline."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from kuai_rank.data import day_boundary, file_sha256, save_json
from kuai_rank.integrity import dataset_snapshot, verify_dataset, check_files
from kuai_rank.retrieval import DualTower, multi_positive_loss, itemcf_graph, build_indexes
from kuai_rank.retrieval_metrics import targets_for_day, exact_validation, ranking_metrics, mean_metrics
from kuai_rank.snapshot import history_before, build_snapshot, FeatureSnapshot
from kuai_rank.pipeline import RecommendationPipeline, blend_order, diversify


def train_retrieval(snapshot, targets, root, config):
    root = Path(root)
    if (root / "retrieval.json").exists():
        meta = json.loads((root / "retrieval.json").read_text())
        if meta["config"] != config:
            raise ValueError("Completed retrieval config differs")
        check_files(root, meta["files"])
        return meta
    pending = root / "training_pending.json"
    if pending.exists():
        meta = json.loads(pending.read_text())
        if meta["config"] != config:
            raise ValueError("Pending retrieval config differs")
        check_files(root, meta["trained_files_sha256"])
        build_indexes(root, np.load(root / "item_vectors.npy", allow_pickle=False))
        meta["files"] = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file() and p.name != "retrieval.json"}
        save_json(root / "retrieval.json", meta)
        return meta
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    graph = snapshot.positives
    pairs = graph.tocoo()
    choices = rng.choice(len(pairs.data), min(config["max_positive_pairs"], len(pairs.data)), replace=False)
    u, i = pairs.row[choices], pairs.col[choices]
    model = DualTower(len(snapshot.user_ids), len(snapshot.video_ids), config["dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    curves, best, best_state = [], -1., None
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        order, total = rng.permutation(len(u)), 0.
        for start in range(0, len(order), config["batch_size"]):
            idx = order[start:start + config["batch_size"]]
            users, items = torch.from_numpy(u[idx].astype(np.int64) + 1), torch.from_numpy(i[idx].astype(np.int64))
            logits = model.user_vectors(users) @ model.item_vectors(items).T / config["temperature"]
            mask = torch.from_numpy(graph[u[idx]][:, i[idx]].toarray().astype(bool))
            loss = multi_positive_loss(logits, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        valid = exact_validation(model, snapshot, targets)
        curves.append({"epoch": epoch + 1, "loss": total / len(u), **valid})
        print("Dual tower", json.dumps(curves[-1]), flush=True)
        if valid["recall@100"] > best:
            best, best_state = valid["recall@100"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), root / "dual_tower.pt")
    with torch.inference_mode():
        vectors = model.item_vectors(torch.arange(len(snapshot.video_ids))).numpy()
    np.save(root / "item_vectors.npy", vectors)
    print("Building blockwise sparse ItemCF and FAISS", flush=True)
    sparse.save_npz(root / "itemcf.npz", itemcf_graph(graph, config["itemcf_neighbors"]))
    meta = {"config": config, "dim": config["dim"], "user_ids": snapshot.user_ids, "video_ids": snapshot.video_ids,
            "training_positive_pairs": len(u), "available_positive_pairs": graph.nnz,
            "snapshot_cutoff_ms": snapshot.meta["cutoff_ms"], "curves": curves,
            "best_validation_recall@100": best, "seconds": time.perf_counter() - started,
            "trained_files_sha256": {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}}
    save_json(pending, meta)
    build_indexes(root, vectors)
    meta["files"] = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}
    save_json(root / "retrieval.json", meta)
    return meta


def evaluate(pipeline, targets, tune=False):
    rows = {route: [] for route in ["popular", "itemcf", "dual_tower", "fusion"]}
    reranked, diverse = [], []
    timings, all_recs, authors = [], set(), set()
    weights = [0., .25, .5, .75, 1.] if tune else [pipeline.rank_weight]
    blends = {str(w): {"uncapped": [], "capped": [], "videos": set(), "authors": set()} for w in weights}
    video_index = {v: i for i, v in enumerate(pipeline.features.video_ids)}
    for count, (user, positive) in enumerate(targets.items()):
        lists, _ = pipeline.retrieval.routes(user, pipeline.recall_k)
        for route, ids in lists.items():
            metrics = ranking_metrics(ids, positive, pipeline.recall_k)
            metrics.update(ranking_metrics(ids, positive, 10))
            rows[route].append(metrics)
        result = pipeline.recommend(user, timestamp_ms=pipeline.features.meta["cutoff_ms"])
        candidates = lists[pipeline.route]
        frame = pipeline.features.features(user, candidates, pipeline.features.meta["cutoff_ms"])
        scores = pipeline.ranker.predict(frame)[:, 0]
        for weight in weights:
            blended_order = blend_order(scores, weight)
            capped = diversify(blended_order, frame.author_id.to_numpy(), 10, pipeline.author_cap)
            blend = blends[str(weight)]
            blend["uncapped"].append(ranking_metrics(candidates[blended_order[:10]], positive, 10))
            blend["capped"].append(ranking_metrics(candidates[capped], positive, 10))
            blend["videos"].update(map(int, candidates[capped]))
            blend["authors"].update(frame.author_id.iloc[capped].astype(str))
        order = blend_order(scores, pipeline.rank_weight)[:10]
        reranked.append(ranking_metrics(candidates[order], positive, 10))
        ids = [video_index[r["video_id"]] for r in result["ranking"]]
        diverse.append(ranking_metrics(ids, positive, 10))
        timings.append(result["timing_ms"]["total"])
        all_recs.update(ids)
        authors.update(r["author_id"] for r in result["ranking"])
        if count % 200 == 0:
            print(f"Evaluated {count + 1}/{len(targets)} users", flush=True)
    blend_metrics = {weight: {"without_author_cap": mean_metrics(v["uncapped"]),
                              "with_author_cap": mean_metrics(v["capped"]),
                              "unique_videos": len(v["videos"]), "unique_authors": len(v["authors"])}
                     for weight, v in blends.items()}
    return {"retrieval": {route: mean_metrics(values) for route, values in rows.items()},
            "ranking_blends": blend_metrics,
            "selected_route": pipeline.route, "ranked_without_author_cap": mean_metrics(reranked),
            "ranked_with_author_cap": mean_metrics(diverse), "unique_recommended_videos": len(all_recs),
            "unique_recommended_authors": len(authors),
            "catalog_recommendation_coverage": len(all_recs) / len(pipeline.features.video_ids),
            "offline_pipeline_ms": dict(zip(["p50", "p95", "p99"], map(float, np.quantile(timings, [.5, .95, .99])))),
            "note": "Metrics judge observed positives only; unexposed videos have unknown outcomes. Catalog is Pure's training-known pool."}


def ann_audit(pipeline, users):
    engine = pipeline.retrieval
    queries = np.concatenate([engine.query_vector(u) for u in list(users)[:200]])
    start = time.perf_counter()
    _, exact = engine.exact.search(queries, 100)
    exact_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    _, approximate = engine.index.search(queries, 100)
    approx_ms = (time.perf_counter() - start) * 1000
    return {"queries": len(queries), "k": 100,
            "neighbor_overlap_with_exact": float(np.mean([len(set(a) & set(b)) / 100 for a, b in zip(exact, approximate)])),
            "flat_batch_ms": exact_ms, "hnsw_batch_ms": approx_ms,
            "scope": "Index fidelity vs exact vector neighbors before seen filtering; not recommendation recall."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/end_to_end.json")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--raw", default="data/raw/KuaiRand-Pure/data")
    parser.add_argument("--out", default="artifacts/end_to_end")
    parser.add_argument("--report", default="results/end_to_end.json")
    args = parser.parse_args()
    if Path(args.report).exists():
        raise ValueError("Completed pipeline report exists; choose new output/report paths")
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    report_root = Path(args.report).parent
    print("Preparing prior-day snapshots and loading existing trained artifacts", flush=True)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    hashes = dataset_snapshot(args.data)
    rank_meta = json.loads((Path(config["ranker"]) / "metrics.json").read_text())
    verify_dataset(args.data, rank_meta)
    metadata = pd.read_csv(Path(args.raw) / "video_features_basic_pure.csv")
    cutoff = day_boundary(config["train_cutoff"])
    history = history_before(args.data, cutoff)
    if not (root / "valid_snapshot/snapshot.json").exists():
        build_snapshot(history, metadata, cutoff, root / "valid_snapshot")
    valid_snapshot = FeatureSnapshot(root / "valid_snapshot")
    valid = pd.read_parquet(Path(args.data) / "valid.parquet")
    targets, valid_coverage = targets_for_day(valid, valid_snapshot, config["eval_users"], config["seed"])
    meta = train_retrieval(valid_snapshot, targets, root / "retrieval", config)
    save_json(report_root / "retrieval_training.json", {k: v for k, v in meta.items() if k not in {"user_ids", "video_ids"}})
    pipeline = RecommendationPipeline(root / "valid_snapshot", root / "retrieval", config["ranker"],
                                      recall_k=config["recall_k"], author_cap=config["author_cap"])
    print("Evaluating validation pipeline", flush=True)
    valid_result = evaluate(pipeline, targets, tune=True)
    selected = max(valid_result["retrieval"], key=lambda route: valid_result["retrieval"][route][f"recall@{config['recall_k']}"])
    if selected != pipeline.route:
        pipeline.route = selected
        valid_result = evaluate(pipeline, targets, tune=True)
    weight = float(max(valid_result["ranking_blends"], key=lambda w: valid_result["ranking_blends"][w]["with_author_cap"]["ndcg@10"]))
    chosen = valid_result["ranking_blends"][str(weight)]
    valid_result["ranked_without_author_cap"] = chosen["without_author_cap"]
    valid_result["ranked_with_author_cap"] = chosen["with_author_cap"]
    valid_result["ranking_model_weight"] = weight
    valid_result["unique_recommended_videos"] = chosen["unique_videos"]
    valid_result["unique_recommended_authors"] = chosen["unique_authors"]
    valid_result["catalog_recommendation_coverage"] = chosen["unique_videos"] / len(valid_snapshot.video_ids)
    # freeze is saved before reading the new retrieval test outcomes.
    freeze = {"config": config, "selected_route": selected, "ranking_model_weight": weight,
              "selection_metric": f"validation recall@{config['recall_k']}, then capped NDCG@10 for ranking blend",
              "prepared_split_sha256": hashes, "ranker_metrics_sha256": file_sha256(Path(config["ranker"]) / "metrics.json"),
              "retrieval_metadata_sha256": file_sha256(root / "retrieval/retrieval.json"),
              "source_sha256": {name: file_sha256(Path("src/kuai_rank") / name) for name in
                                  ["snapshot.py", "retrieval.py", "pipeline.py", "retrieval_metrics.py", "inference.py"]},
              "scope": "Extension test reuses the previously reported ranking holdout; not a fresh blind dataset."}
    if (root / "freeze.json").exists():
        if json.loads((root / "freeze.json").read_text()) != freeze:
            raise ValueError("Pipeline, sources or dataset changed after test freeze")
    else:
        save_json(root / "freeze.json", freeze)
    save_json(report_root / "end_to_end_freeze.json", freeze)
    test_cutoff = day_boundary(config["test_cutoff"])
    if not (root / "test_snapshot/snapshot.json").exists():
        build_snapshot(history_before(args.data, test_cutoff), metadata, test_cutoff, root / "test_snapshot", valid_snapshot.video_ids)
    test_pipeline = RecommendationPipeline(root / "test_snapshot", root / "retrieval", config["ranker"], selected,
                                           config["recall_k"], config["author_cap"], weight)
    test = pd.read_parquet(Path(args.data) / "test.parquet")
    test_targets, test_coverage = targets_for_day(test, test_pipeline.features, config["eval_users"], config["seed"])
    print("Evaluating frozen test pipeline", flush=True)
    test_result = evaluate(test_pipeline, test_targets)
    if dataset_snapshot(args.data) != hashes:
        raise ValueError("Dataset changed during pipeline experiment")
    result = {"config": config, "catalog_videos": len(valid_snapshot.video_ids), "selected_route": selected, "ranking_model_weight": weight,
              "validation_coverage": valid_coverage, "test_coverage": test_coverage,
              "validation": valid_result, "test": test_result, "ann_audit": ann_audit(test_pipeline, test_targets),
              "freeze_sha256": file_sha256(root / "freeze.json"),
              "limitations": ["Pure retains incomplete candidate-pool histories, not all-platform videos.",
                              "Only first validation/test days and a fixed sample of eligible users are evaluated.",
                              "Observed-positive retrieval metrics are not fully judged catalog ranking or causal policy utility.",
                              "User/item/author features and profiles replay prior standard logs; no model-policy feedback simulation.",
                              "Candidate duration uses pre-cutoff median, while original exposure ranker used per-event duration.",
                              "One retrieval initialization; the original ranking experiment uses three neural seeds.",
                              "Previously reported test is reused for extension reporting; no claim of a new blind holdout."]}
    save_json(args.report, result)
    first_user = next(iter(test_targets))
    save_json(report_root / "end_to_end_example.json", test_pipeline.recommend(first_user))
    save_json(root / "deployment.json", {"snapshot": str(root / "test_snapshot"), "retrieval": str(root / "retrieval"),
              "ranker": config["ranker"], "route": selected, "recall_k": config["recall_k"], "author_cap": config["author_cap"], "rank_weight": weight})
    print(json.dumps({"selected_route": selected, "test": test_result, "ann_audit": result["ann_audit"]}, indent=2), flush=True)

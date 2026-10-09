import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from kuai_rank.data import day_boundary, build_features, file_sha256
from kuai_rank.encoding import FeatureEncoder
from kuai_rank.models import RankModel
from kuai_rank.pipeline import diversify, RecommendationPipeline, blend_order
from kuai_rank.retrieval import DualTower, multi_positive_loss, top_indices, rrf, itemcf_graph, build_indexes
from kuai_rank.retrieval_metrics import targets_for_day, ranking_metrics
from kuai_rank.snapshot import build_snapshot, FeatureSnapshot


def fixture():
    cutoff = day_boundary("2022-04-22")
    frame = pd.DataFrame({"user_id": [1, 1, 2, 1, 1], "video_id": [10, 20, 10, 20, 30],
                          "author_id": [100, 200, 100, 200, 300], "time_ms": [cutoff - 1000] * 3 + [cutoff - 1000, cutoff],
                          "traffic": ["standard"] * 3 + ["random", "standard"],
                          "long_view": [1, 0, 1, 1, 1], "is_click": [1, 0, 1, 1, 1], "is_like": [0, 0, 1, 1, 1],
                          "duration_log": [np.log1p(10)] * 5, "duration_ms": [10000] * 5, "tab": [0] * 5})
    meta = pd.DataFrame({"video_id": [10, 20, 30], "author_id": [100, 200, 300],
                         "video_type": ["NORMAL"] * 3, "upload_type": ["ShortImport"] * 3,
                         "music_type": [1] * 3, "tag": ["1,2"] * 3, "upload_dt": ["2022-04-01"] * 3,
                         "server_width": [720] * 3, "server_height": [1280] * 3})
    return cutoff, frame, meta


def test_snapshot_filters_future_and_random_feedback_and_preserves_feature_formulas(tmp_path):
    cutoff, frame, meta = fixture()
    build_snapshot(frame, meta, cutoff, tmp_path / "snapshot")
    snapshot = FeatureSnapshot(tmp_path / "snapshot")
    assert snapshot.video_ids == ["10", "20"]
    assert snapshot.meta["max_history_ms"] < cutoff
    assert len(snapshot.seen_indices("1")) == 2
    assert len(snapshot.positive_indices("1")) == 1
    request = snapshot.features("1", [0], cutoff, "0")
    assert np.isclose(request.user_count_log.iloc[0], np.log1p(2))
    assert np.isclose(request.user_long_view_rate.iloc[0], (1 + 3) / 12)
    # Compare independent serving features with the original offline replay.
    earlier = frame.iloc[:3].drop(columns=["author_id", "duration_log"]).copy()
    event = earlier.iloc[:1].copy()
    event["time_ms"] = cutoff
    event["traffic"] = "random"
    expected = build_features(pd.concat([earlier, event], ignore_index=True), meta).iloc[-1]
    for feature in ["duration_log", "aspect_ratio", "video_age_log", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                    "user_count_log", "user_long_view_rate", "item_count_log", "item_is_like_rate", "author_count_log"]:
        assert np.isclose(request[feature].iloc[0], expected[feature]), feature
    with pytest.raises(ValueError, match="snapshot date"):
        snapshot.features("1", [0], cutoff + 86400000)
    cold = snapshot.features("cold", [0], cutoff)
    assert cold.user_count_log.iloc[0] == 0
    assert cold.user_long_view_rate.iloc[0] == .3


def test_snapshot_file_tampering_is_rejected(tmp_path):
    cutoff, frame, meta = fixture()
    root = tmp_path / "snapshot"
    build_snapshot(frame, meta, cutoff, root)
    with (root / "popularity.npy").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="changed since training"):
        FeatureSnapshot(root)


def test_recall_filters_seen_items_and_rrf_keeps_unique_provenance():
    np.testing.assert_array_equal(top_indices([3., 2., 1.], 10, [0]), [1, 2])
    np.testing.assert_array_equal(top_indices([3, 2, 1], 10, [0]), [1, 2])
    ids, sources = rrf({"a": [1, 2], "b": [2, 3]}, 10)
    assert len(ids) == len(set(ids)) == 3
    assert ids[0] == 2
    assert sources[2] == ["a", "b"]


def test_itemcf_uses_cosine_neighbors_and_excludes_self():
    graph = sparse.csr_matrix([[1., 1., 0.], [1., 0., 1.]], dtype=np.float32)
    cf = itemcf_graph(graph, neighbors=2, block=1)
    assert cf.diagonal().sum() == 0
    assert np.isclose(cf[0, 1], 1 / np.sqrt(2))
    assert cf[1, 2] == 0


def test_multi_positive_loss_does_not_treat_duplicate_targets_as_negatives():
    logits = torch.tensor([[2., 2., 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss = multi_positive_loss(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0 and logits.grad[0, 2] > 0
    assert multi_positive_loss(logits, torch.ones_like(mask)).item() == 0
    with pytest.raises(ValueError, match="positive"):
        multi_positive_loss(logits, torch.zeros_like(mask))


def test_target_protocol_excludes_repeated_and_unavailable_future_videos(tmp_path):
    cutoff, frame, meta = fixture()
    root = tmp_path / "snapshot"
    build_snapshot(frame, meta, cutoff, root)
    snapshot = FeatureSnapshot(root)
    future = pd.DataFrame({"user_id": ["1", "2", "2"], "video_id": ["10", "20", "30"],
                           "time_ms": [cutoff] * 3, "long_view": [1] * 3})
    targets, audit = targets_for_day(future, snapshot)
    assert targets == {"2": {1}}
    assert audit["previously_seen_pairs_excluded"] == 1
    assert audit["outside_catalog_pairs"] == 1
    values = ranking_metrics([0, 1], {1}, 10)
    assert values["recall@10"] == 1
    assert np.isclose(values["ndcg@10"], 1 / np.log2(3))


def test_author_cap_is_strict_even_if_result_list_becomes_short():
    result = diversify([0, 1, 2, 3], ["a", "a", "a", "b"], k=4, author_cap=2)
    np.testing.assert_array_equal(result, [0, 1, 3])


def test_ranking_blend_preserves_retrieval_and_model_endpoints():
    np.testing.assert_array_equal(blend_order([.1, .9, .3], 0), [0, 1, 2])
    np.testing.assert_array_equal(blend_order([.1, .9, .3], 1), [1, 2, 0])
    with pytest.raises(ValueError, match="rank_weight"):
        blend_order([.1], 2)


def test_full_pipeline_can_recommend_to_known_and_cold_users_without_supplied_features(tmp_path):
    pytest.importorskip("faiss")
    cutoff, frame, metadata = fixture()
    snapshot_dir, retrieval_dir, rank_dir = tmp_path / "snapshot", tmp_path / "retrieval", tmp_path / "rank"
    build_snapshot(frame, metadata, cutoff, snapshot_dir)
    snapshot = FeatureSnapshot(snapshot_dir)
    retrieval_dir.mkdir()
    model = DualTower(2, 2, dim=4)
    torch.save(model.state_dict(), retrieval_dir / "dual_tower.pt")
    with torch.inference_mode():
        vectors = model.item_vectors(torch.arange(2)).numpy()
    build_indexes(retrieval_dir, vectors)
    sparse.save_npz(retrieval_dir / "itemcf.npz", itemcf_graph(snapshot.positives))
    retrieval = {"user_ids": snapshot.user_ids, "video_ids": snapshot.video_ids, "dim": 4,
                 "files": {p.name: file_sha256(p) for p in retrieval_dir.iterdir()}}
    (retrieval_dir / "retrieval.json").write_text(json.dumps(retrieval))
    rank_dir.mkdir()
    features = snapshot.features("cold", [0, 1], cutoff)
    encoder = FeatureEncoder(["user_id", "video_id"], ["duration_log", "user_count_log"]).fit(features)
    with (rank_dir / "encoder.pkl").open("wb") as output:
        pickle.dump(encoder, output)
    ranker = RankModel(encoder.sizes, 2, embedding_dim=4, hidden_dim=8)
    torch.save(ranker.state_dict(), rank_dir / "checkpoint.pt")
    rank_meta = {"model": "deepfm", "tasks": ["long_view"], "config": {"threads": 1, "embedding_dim": 4,
                 "hidden_dim": 8, "experts": 4, "tasks": ["long_view"]},
                 "model_artifact_sha256": {p.name: file_sha256(p) for p in rank_dir.iterdir()}}
    (rank_dir / "metrics.json").write_text(json.dumps(rank_meta))
    pipeline = RecommendationPipeline(snapshot_dir, retrieval_dir, rank_dir, recall_k=2)
    cold = pipeline.recommend("cold", k=2, timestamp_ms=cutoff)
    assert cold["cold_user"] and len(cold["ranking"]) == 2
    known = pipeline.recommend("2", k=2, timestamp_ms=cutoff)
    assert [row["video_id"] for row in known["ranking"]] == ["20"]
    assert pipeline.recommend("1", timestamp_ms=cutoff)["ranking"] == []
    # Single-route optimization must preserve the original all-route result.
    for user in ["1", "2", "cold"]:
        routes, _ = pipeline.retrieval.routes(user, 2)
        for route, expected in routes.items():
            actual, _ = pipeline.retrieval.recall(user, 2, route)
            np.testing.assert_array_equal(actual, expected)
    # Frozen weight zero may score only selected rows, but must return exactly
    # the former full-candidate scores, provenance and author-constrained order.
    pipeline.rank_weight = 0
    for user in ["2", "cold"]:
        ids, sources = pipeline.retrieval.recall(user, 2, "fusion")
        full_frame = pipeline.features.features(user, ids, cutoff)
        full_scores = pipeline.ranker.predict(full_frame)
        selected = diversify(np.arange(len(ids)), full_frame.author_id.to_numpy(), 1, 2)
        expected = [{"video_id": str(full_frame.video_id.iloc[i]), "author_id": str(full_frame.author_id.iloc[i]),
                     "scores": dict(zip(pipeline.ranker.meta["tasks"], map(float, full_scores[i]))),
                     "recall_sources": sources[int(ids[i])]} for i in selected]
        optimized = pipeline.recommend(user, k=1, timestamp_ms=cutoff)
        assert optimized["ranking"] == expected
        assert optimized["scored_candidates"] == 1
        assert optimized["candidate_count"] == len(ids)
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps({"snapshot": str(snapshot_dir), "retrieval": str(retrieval_dir), "ranker": str(rank_dir), "recall_k": 2}))
    from kuai_rank.recommend_api import create_app, RecommendationRequest
    from fastapi import HTTPException
    app = create_app(deployment)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/recommend")
    assert endpoint(RecommendationRequest(user_id="cold"))["ranking"]
    with pytest.raises(HTTPException) as error:
        endpoint(RecommendationRequest(user_id="cold", timestamp_ms=0))
    assert error.value.status_code == 422

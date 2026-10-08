import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)

from kuai_rank.data import daily_history, split_labels, validate_logs, day_boundary, TASKS
from kuai_rank.encoding import FeatureEncoder
from kuai_rank.metrics import binary_metrics, grouped_metrics
from kuai_rank.models import RankModel


def logs():
    return pd.DataFrame({"user_id": [1, 1, 1, 1, 1], "video_id": [3] * 5,
                         "time_ms": [1, 1, 2, 3, 4], "day": [0, 0, 1, 1, 2],
                         "tab": [0] * 5, "duration_ms": [1000] * 5,
                         "long_view": [1, 0, 1, 1, 1], "is_click": [1, 0, 1, 1, 1],
                         "is_like": [0, 0, 1, 1, 1],
                         "traffic": ["standard", "standard", "random", "standard", "standard"]})


def test_day_ties_and_current_labels_cannot_leak():
    frame = logs()
    original = daily_history(frame, "user_id", "user")
    assert original["user_count_log"][0] == 0
    assert original["user_count_log"][1] == 0
    assert np.isclose(original["user_count_log"][2], np.log1p(2))
    changed = frame.copy()
    changed.loc[changed.day == 1, TASKS] = 0
    modified = daily_history(changed, "user_id", "user")
    for feature in original:
        np.testing.assert_allclose(original[feature][:4], modified[feature][:4])


def test_random_exposures_do_not_update_history():
    frame = logs()
    values = daily_history(frame, "user_id", "user")
    assert np.isclose(values["user_count_log"][4], np.log1p(3))
    frame.loc[frame.traffic == "random", TASKS] = 0
    changed = daily_history(frame, "user_id", "user")
    for feature in values:
        np.testing.assert_allclose(values[feature], changed[feature])


def test_future_labels_cannot_change_earlier_features():
    frame = logs()
    before = daily_history(frame, "video_id", "item")
    frame.loc[frame.day == 2, TASKS] = 0
    after = daily_history(frame, "video_id", "item")
    for feature in before:
        np.testing.assert_allclose(before[feature], after[feature])


def test_history_respects_entity_boundaries_and_input_order():
    frame = logs().sample(frac=1, random_state=1).reset_index(drop=True)
    frame.loc[0, "user_id"] = 999
    actual = daily_history(frame, "user_id", "user")
    assert actual["user_count_log"][0] == 0
    for i, row in frame.iterrows():
        prior = frame[(frame.day < row.day) & (frame.user_id == row.user_id) & (frame.traffic == "standard")]
        assert np.isclose(actual["user_count_log"][i], np.log1p(len(prior)))


def test_split_boundaries_are_half_open_and_policies_separate():
    config = {"train_end": "2022-04-18", "valid_end": "2022-04-22",
              "random_calibration_end": "2022-04-26", "random_valid_end": "2022-05-01"}
    dates = ["2022-04-17", "2022-04-18", "2022-04-22", "2022-04-25", "2022-04-26", "2022-05-01"]
    frame = pd.DataFrame({"time_ms": [day_boundary(d) for d in dates],
                          "traffic": ["standard"] * 3 + ["random"] * 3})
    assert list(split_labels(frame, config)) == ["train", "valid", "test", "random_calibration", "random_valid", "random_test"]


def test_unknown_categories_do_not_expand_training_vocabulary():
    encoder = FeatureEncoder(["cat"], ["num"]).fit(pd.DataFrame({"cat": ["a", "b"], "num": [1, 3]}))
    cats, nums = encoder.transform(pd.DataFrame({"cat": ["c"], "num": [2]}))
    assert cats[0, 0] == 0
    assert nums[0, 0] == 0
    assert encoder.sizes == [3]


def test_gauc_reports_only_eligible_users():
    values = grouped_metrics([0, 1, 1, 1], [.1, .9, .2, .3], [1, 1, 2, 2], [0] * 4)
    assert values["user_gauc"] == 1
    assert values["gauc_eligible_users"] == 1
    assert values["gauc_row_coverage"] == .5


def test_ndcg_matches_hand_computed_exposure_ranking():
    values = grouped_metrics([1, 0, 1], [.3, .9, .2], [1] * 3, [0] * 3, k=2)
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert np.isclose(values["logged_user_day_ndcg@2"], expected)


def test_single_class_auc_is_undefined_not_fabricated():
    result = binary_metrics([0, 0], [.2, .3])
    assert result["auc"] is None
    assert result["logloss"] > 0


def test_nonbinary_labels_rejected():
    frame = logs()
    frame.loc[0, "is_like"] = 2
    with pytest.raises(ValueError, match="Nonbinary"):
        validate_logs(frame)


@pytest.mark.parametrize("kind,tasks", [("deepfm", 1), ("sharedbottom", 3), ("mmoe", 3)])
def test_models_produce_finite_gradients_for_all_used_parameters(kind, tasks):
    torch.manual_seed(7)
    model = RankModel([5, 8], 3, kind=kind, embedding_dim=4, hidden_dim=16)
    cats = torch.tensor([[1, 2], [3, 4]])
    nums = torch.randn(2, 3)
    logits = model(cats, nums)
    assert logits.shape == (2, tasks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if kind == "mmoe":
        np.testing.assert_allclose(model.gate_mean(cats, nums).detach().sum(1), 1, atol=1e-6)


def test_fm_interactions_equal_pairwise_dot_product():
    model = RankModel([2, 2], 1, embedding_dim=2, hidden_dim=8)
    with torch.no_grad():
        model.embeddings[0].weight[1] = torch.tensor([1., 2.])
        model.embeddings[1].weight[1] = torch.tensor([3., 4.])
        model.linear_numeric.weight.zero_()
        model.linear_numeric.bias.zero_()
    _, base = model.representation(torch.tensor([[1, 1]]), torch.zeros(1, 1))
    assert base.item() == 11.0


def test_gauc_uses_average_ranks_for_tied_predictions():
    values = grouped_metrics([0, 1, 0, 1], [.5, .5, .5, .5], [1] * 4, [0] * 4)
    assert values["user_gauc"] == .5


def test_real_schema_preparation_preserves_splits_and_excludes_unsafe_features(tmp_path):
    import json
    from pathlib import Path
    from kuai_rank.data import prepare, BASIC_COLUMNS
    config = json.loads((Path(__file__).parents[1] / "configs/pure.json").read_text())
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    dates = ["2022-04-21", "2022-04-22", "2022-05-01", "2022-04-23", "2022-04-27", "2022-05-02"]
    frame = pd.DataFrame({"user_id": [1] * 6, "video_id": [3] * 6,
                          "time_ms": [day_boundary(d) for d in dates], "tab": [0] * 6,
                          "duration_ms": [1000] * 6, "long_view": [1, 0, 1, 0, 1, 0],
                          "is_click": [1, 0, 1, 0, 1, 0], "is_like": [0] * 6,
                          "play_time_ms": [999999] * 6})
    frame.iloc[:3].to_csv(raw / "log_standard_fixture.csv", index=False)
    frame.iloc[3:].to_csv(raw / "log_random_fixture.csv", index=False)
    basic = {"video_id": 3, "author_id": 4, "video_type": "NORMAL", "upload_type": "ShortImport",
             "music_type": 1, "tag": "1,2", "upload_dt": "2022-04-01", "server_width": 720, "server_height": 1280}
    pd.DataFrame([basic], columns=BASIC_COLUMNS).to_csv(raw / "video_features_basic_fixture.csv", index=False)
    prepare(raw, out, config)
    valid = pd.read_parquet(out / "valid.parquet")
    assert len(valid) == 1
    assert np.isclose(valid.user_count_log.iloc[0], np.log1p(1))
    assert "play_time_ms" not in valid.columns
    assert "show_cnt" not in valid.columns
    audit = json.loads((out / "audit.json").read_text())
    assert all(part["rows"] == 1 for part in audit["splits"].values())


def test_saved_neural_ranker_matches_training_inference(tmp_path):
    import json
    import pickle
    from kuai_rank.serving import create_app, RankRequest, Candidate
    from kuai_rank.train import predict_neural
    from kuai_rank.data import file_sha256
    from kuai_rank.inference import Scorer
    from fastapi import HTTPException
    config = {"embedding_dim": 4, "hidden_dim": 16, "experts": 4, "threads": 1,
              "tasks": ["long_view", "is_click", "is_like"]}
    frame = pd.DataFrame({"cat": ["a", "b"], "num": [1., 3.]})
    encoder = FeatureEncoder(["cat"], ["num"]).fit(frame)
    with (tmp_path / "encoder.pkl").open("wb") as f:
        pickle.dump(encoder, f)
    model = RankModel(encoder.sizes, 1, kind="mmoe", embedding_dim=4, hidden_dim=16)
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    meta = {"model": "mmoe", "tasks": config["tasks"], "config": config, "data_manifest_sha256": "fixture",
            "model_artifact_sha256": {"checkpoint.pt": file_sha256(tmp_path / "checkpoint.pt"),
                                      "encoder.pkl": file_sha256(tmp_path / "encoder.pkl")}}
    (tmp_path / "metrics.json").write_text(json.dumps(meta))
    app = create_app(tmp_path)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/rank")
    response = endpoint(RankRequest(candidates=[Candidate(candidate_id=str(i), categorical={"cat": row["cat"]},
                       numeric={"num": row["num"]}) for i, row in frame.iterrows()], k=2))
    c, n = encoder.transform(frame)
    expected = predict_neural(model, c, n)
    for item in response["ranking"]:
        np.testing.assert_allclose(list(item["scores"].values()), expected[int(item["candidate_id"])], atol=1e-6)
    assert response["ranking"][0]["scores"]["long_view"] >= response["ranking"][1]["scores"]["long_view"]
    duplicate = Candidate(candidate_id="same", categorical={"cat": "a"}, numeric={"num": 1.})
    with pytest.raises(HTTPException, match="Duplicate"):
        endpoint(RankRequest(candidates=[duplicate, duplicate]))
    with pytest.raises(HTTPException, match="contract mismatch"):
        endpoint(RankRequest(candidates=[Candidate(candidate_id="x", categorical={"cat": "a"}, numeric={"wrong": 1.})]))
    with (tmp_path / "checkpoint.pt").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="changed since training"):
        Scorer(tmp_path)


def test_config_rejects_current_outcomes_as_features():
    import json
    from pathlib import Path
    from kuai_rank.data import validate_config
    config = json.loads((Path(__file__).parents[1] / "configs/pure.json").read_text())
    config["numeric"].append("long_view")
    with pytest.raises(ValueError, match="outcome-leaking"):
        validate_config(config)


def test_capacity_matched_baseline_controls_total_parameter_budget():
    from kuai_rank.models import capacity_matched_hidden, parameter_budget
    sizes, n_numeric, dim, hidden, experts = [1000, 500], 19, 12, 96, 4
    h = capacity_matched_hidden(sizes, n_numeric, dim, hidden, experts)
    baseline = RankModel(sizes, n_numeric, "deepfm", dim, h, experts)
    candidate = RankModel(sizes, n_numeric, "mmoe", dim, hidden, experts)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    candidate_count = sum(p.numel() for p in candidate.parameters())
    assert candidate_count == parameter_budget(sizes, n_numeric, "mmoe", dim, hidden, experts)
    assert abs(baseline_count - candidate_count) / candidate_count < .01

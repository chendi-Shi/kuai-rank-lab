import numpy as np
import pytest
import torch

from kuai_rank.data import file_sha256
from kuai_rank.integrity import check_files, dataset_snapshot, verify_dataset, verify_models, verify_predictions, SPLITS
from kuai_rank.models import RankModel


@pytest.mark.parametrize("kind", ["deepfm", "sharedbottom", "mmoe"])
def test_unknown_embedding_stays_neutral_when_training_batch_contains_unknowns(kind):
    model = RankModel([4, 5], 2, kind, embedding_dim=3, hidden_dim=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    cats = torch.tensor([[0, 1], [2, 0], [1, 2]])
    for _ in range(3):
        optimizer.zero_grad()
        model(cats, torch.ones(3, 2)).sum().backward()
        optimizer.step()
    for layer in [*model.embeddings, *model.linear_fields]:
        assert torch.count_nonzero(layer.weight[0]).item() == 0
        assert torch.count_nonzero(layer.weight[1:]).item() > 0


def test_dataset_drift_is_rejected_even_when_manifest_did_not_change(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    for split in SPLITS:
        (tmp_path / f"{split}.parquet").write_bytes(b"original split bytes")
    meta = {"data_manifest_sha256": file_sha256(tmp_path / "manifest.json"),
            "prepared_split_sha256": dataset_snapshot(tmp_path)}
    assert verify_dataset(tmp_path, meta)["prepared_split_bytes_verified"]
    (tmp_path / "random_test.parquet").write_bytes(b"different split bytes")
    with pytest.raises(ValueError, match="random_test"):
        verify_dataset(tmp_path, meta)


def test_legacy_dataset_verification_does_not_claim_missing_byte_coverage(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    meta = {"data_manifest_sha256": file_sha256(tmp_path / "manifest.json")}
    assert verify_dataset(tmp_path, meta) == {"manifest_verified": True, "prepared_split_bytes_verified": False}


def test_incomplete_model_hash_metadata_is_rejected_before_deserialization(tmp_path):
    (tmp_path / "encoder.pkl").write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="fingerprints are missing"):
        verify_models(tmp_path, {"model": "lr", "model_artifact_sha256": {
            "encoder.pkl": file_sha256(tmp_path / "encoder.pkl")}})


def test_changed_prediction_arrays_are_rejected(tmp_path):
    path = tmp_path / "predictions_valid.npz"
    np.savez(path, probabilities=[.1, .8])
    meta = {"predictions": {"valid": {"file": path.name, "sha256": file_sha256(path)}}}
    assert verify_predictions(tmp_path, meta) == 1
    np.savez(path, probabilities=[.2, .8])
    with pytest.raises(ValueError, match="changed since training"):
        verify_predictions(tmp_path, meta)


def test_hash_contract_cannot_escape_run_directory(tmp_path):
    with pytest.raises(ValueError, match="local filename"):
        check_files(tmp_path, {"../outside.pkl": "unused"})

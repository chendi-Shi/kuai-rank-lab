"""Serve a trained ranker over caller-supplied point-in-time candidate features.

The API does not fabricate a retrieval stage or an online feature store. Model
files and pickles must be trusted artifacts produced locally by this project.
"""
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .inference import Scorer


class Candidate(BaseModel):
    candidate_id: str
    categorical: dict[str, str]
    numeric: dict[str, float]


class RankRequest(BaseModel):
    candidates: list[Candidate] = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=500)


def create_app(run_dir=None):
    root = Path(run_dir or os.environ.get("KUAI_RANK_RUN", "results/initial/lightgbm_s2026"))
    scorer = Scorer(root)
    meta, encoder = scorer.meta, scorer.encoder
    kind = meta["model"]
    app = FastAPI(title="KuaiRank candidate ranking", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": kind, "tasks": meta["tasks"],
                "data_manifest_sha256": meta["data_manifest_sha256"]}

    @app.post("/rank")
    def rank(request: RankRequest):
        start = time.perf_counter()
        identifiers = [candidate.candidate_id for candidate in request.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise HTTPException(422, "Duplicate candidate IDs")
        rows = []
        for candidate in request.candidates:
            if set(candidate.categorical) != set(encoder.categorical) or set(candidate.numeric) != set(encoder.numeric):
                raise HTTPException(422, "Feature contract mismatch; use the run's categorical and numeric feature lists")
            if not np.isfinite(list(candidate.numeric.values())).all():
                raise HTTPException(422, "Numeric features must be finite")
            rows.append({**candidate.categorical, **candidate.numeric})
        frame = pd.DataFrame(rows)
        predictions = scorer.predict(frame)
        order = np.argsort(-predictions[:, 0], kind="stable")[:request.k]
        return {"model": kind, "objective": meta["tasks"][0], "ranking": [
            {"candidate_id": identifiers[i], "scores": dict(zip(meta["tasks"], map(float, predictions[i])))}
            for i in order], "ranking_ms": (time.perf_counter() - start) * 1000,
            "unknown_category_rates": encoder.unknown_rates(frame)}

    return app

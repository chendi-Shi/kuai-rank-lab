"""Full recommendation API using immutable historical feature snapshots."""
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .pipeline import RecommendationPipeline


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    k: int = Field(default=10, ge=1, le=50)
    tab: str = "0"
    timestamp_ms: int | None = None
    route: Literal["popular", "itemcf", "dual_tower", "fusion"] | None = None


def create_app(deployment=None):
    deployment = Path(deployment or os.environ.get("KUAI_DEPLOYMENT", "artifacts/end_to_end/deployment.json"))
    config = json.loads(deployment.read_text(encoding="utf-8"))
    pipeline = RecommendationPipeline(**config)
    app = FastAPI(title="KuaiRank full recommendation pipeline", version="0.2.0")

    @app.get("/", response_class=HTMLResponse)
    def demo():
        return Path(__file__).with_name("demo.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "catalog_videos": len(pipeline.features.video_ids),
                "snapshot_ms": pipeline.features.meta["cutoff_ms"], "route": pipeline.route,
                "ranker": pipeline.ranker.meta["model"], "demo_user": pipeline.features.user_ids[0],
                "ranking_model_weight": pipeline.rank_weight,
                "scope": "historical Pure-pool demonstration"}

    @app.post("/recommend")
    def recommend(request: RecommendationRequest):
        try:
            return pipeline.recommend(request.user_id, request.k, request.tab, request.timestamp_ms, request.route)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app

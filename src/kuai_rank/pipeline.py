"""A user-only request traverses recall, features, ranking and diversification."""
import time

import numpy as np

from .inference import Scorer
from .retrieval import RecallEngine
from .snapshot import FeatureSnapshot


def diversify(indices, authors, k, author_cap=2):
    counts, selected = {}, []
    for i in indices:
        author = str(authors[i])
        if counts.get(author, 0) >= author_cap:
            continue
        selected.append(int(i))
        counts[author] = counts.get(author, 0) + 1
        if len(selected) == k:
            break
    # Strict cap: return fewer results rather than silently violate the rule.
    return np.asarray(selected, dtype=np.int64)


def blend_order(probabilities, rank_weight=1.):
    if not 0 <= rank_weight <= 1:
        raise ValueError("rank_weight must lie in [0, 1]")
    n = len(probabilities)
    if not n:
        return np.array([], dtype=np.int64)
    learned_order = np.argsort(-np.asarray(probabilities), kind="stable")
    learned_ranks = np.empty(n, dtype=np.int64)
    learned_ranks[learned_order] = np.arange(n)
    scores = (1 - rank_weight) * (1 - np.arange(n) / n) + rank_weight * (1 - learned_ranks / n)
    return np.argsort(-scores, kind="stable")


class RecommendationPipeline:
    def __init__(self, snapshot, retrieval, ranker, route="fusion", recall_k=300, author_cap=2, rank_weight=1.):
        self.features = FeatureSnapshot(snapshot)
        self.retrieval = RecallEngine(retrieval, self.features)
        self.ranker = Scorer(ranker)
        self.route, self.recall_k, self.author_cap = route, recall_k, author_cap
        self.rank_weight = rank_weight

    def recommend(self, user, k=10, tab="0", timestamp_ms=None, route=None):
        start = time.perf_counter()
        ids, sources = self.retrieval.recall(user, self.recall_k, route or self.route)
        candidate_count = len(ids)
        # With a frozen zero model weight, order is known before scoring.
        # Keep the exact author constraint and score only returned videos.
        if self.rank_weight == 0 and len(ids):
            authors = self.features.catalog.author_id.to_numpy()[ids]
            selected = diversify(np.arange(len(ids)), authors, k, self.author_cap)
            ids = ids[selected]
        recalled = time.perf_counter()
        frame = self.features.features(user, ids, timestamp_ms, tab)
        featured = time.perf_counter()
        if len(ids) == 0:
            return {"user_id": str(user), "snapshot_ms": self.features.meta["cutoff_ms"],
                    "route": route or self.route, "candidate_count": 0,
                    "ranking_model_weight": self.rank_weight,
                    "cold_user": str(user) not in self.retrieval.user_index, "ranking": [],
                    "timing_ms": {"recall": (recalled - start) * 1000,
                                  "features": (featured - recalled) * 1000,
                                  "ranking_and_diversity": 0., "total": (featured - start) * 1000},
                    "unknown_category_rates": {}}
        probabilities = self.ranker.predict(frame)
        order = blend_order(probabilities[:, 0], self.rank_weight)
        selected = diversify(order, frame.author_id.to_numpy(), k, self.author_cap)
        finished = time.perf_counter()
        return {"user_id": str(user), "snapshot_ms": self.features.meta["cutoff_ms"],
                "route": route or self.route, "candidate_count": candidate_count,
                "scored_candidates": len(ids),
                "unknown_category_rate_scope": "returned_videos" if self.rank_weight == 0 else "all_candidates",
                "ranking_model_weight": self.rank_weight,
                "cold_user": str(user) not in self.retrieval.user_index,
                "ranking": [{"video_id": str(frame.video_id.iloc[i]), "author_id": str(frame.author_id.iloc[i]),
                             "scores": dict(zip(self.ranker.meta["tasks"], map(float, probabilities[i]))),
                             "recall_sources": sources[int(ids[i])]} for i in selected],
                "timing_ms": {"recall": (recalled - start) * 1000, "features": (featured - recalled) * 1000,
                              "ranking_and_diversity": (finished - featured) * 1000, "total": (finished - start) * 1000},
                "unknown_category_rates": self.ranker.encoder.unknown_rates(frame)}

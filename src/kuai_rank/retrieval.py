"""Popular / sparse ItemCF / learned dual-tower retrieval and FAISS indexes."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from .data import file_sha256, save_json
from .integrity import check_files


class DualTower(nn.Module):
    """ID-based two-tower baseline, with independent normalized MLP towers."""
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.users = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.items = nn.Embedding(n_items, dim)
        self.user_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim))
        self.item_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim))
        nn.init.normal_(self.users.weight, std=.05)
        nn.init.normal_(self.items.weight, std=.05)
        with torch.no_grad():
            self.users.weight[0].zero_()

    def user_vectors(self, ids):
        return F.normalize(self.user_mlp(self.users(ids)), dim=1)

    def item_vectors(self, ids):
        return F.normalize(self.item_mlp(self.items(ids)), dim=1)


def multi_positive_loss(logits, positive_mask):
    # Every known long-view item for a user in the batch is a positive. Repeated
    # item IDs and other known positives must not become accidental negatives.
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every row needs at least one positive")
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    return (torch.logsumexp(logits, dim=1) - numerator).mean()


def itemcf_graph(positives, neighbors=80, block=128):
    norms = np.sqrt(np.asarray(positives.sum(axis=0)).ravel()).clip(min=1)
    normalized = positives.multiply(1 / norms).tocsc()
    rows, cols, scores = [], [], []
    for start in range(0, positives.shape[1], block):
        similarities = (normalized[:, start:start + block].T @ normalized).tocsr()
        for local in range(similarities.shape[0]):
            lo, hi = similarities.indptr[local:local + 2]
            ids, values = similarities.indices[lo:hi], similarities.data[lo:hi]
            keep = (ids != start + local) & (values > 0)
            ids, values = ids[keep], values[keep]
            order = np.lexsort((ids, -values))[:neighbors]
            rows.extend([start + local] * len(order))
            cols.extend(ids[order])
            scores.extend(values[order])
    return sparse.csr_matrix((np.asarray(scores, dtype=np.float32), (rows, cols)),
                             shape=(positives.shape[1], positives.shape[1]))


def top_indices(scores, k, seen=()):
    values = np.asarray(scores, dtype=np.float64).copy()
    values[np.asarray(seen, dtype=np.int64)] = -np.inf
    ids = np.flatnonzero(np.isfinite(values))
    return ids[np.lexsort((ids, -values[ids]))[:k]]


def rrf(lists, k, constant=60):
    scores = {}
    sources = {}
    for name, indices in lists.items():
        for rank, item in enumerate(indices):
            item = int(item)
            scores[item] = scores.get(item, 0.) + 1 / (constant + rank + 1)
            sources.setdefault(item, []).append(name)
    ids = sorted(scores, key=lambda i: (-scores[i], i))[:k]
    return np.asarray(ids, dtype=np.int64), {i: sources[i] for i in ids}


def build_indexes(root, vectors):
    import faiss
    faiss.omp_set_num_threads(1)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    exact = faiss.IndexFlatIP(vectors.shape[1])
    exact.add(vectors)
    approx = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    approx.hnsw.efConstruction = 120
    approx.hnsw.efSearch = 96
    approx.add(vectors)
    faiss.write_index(exact, str(Path(root) / "flat.faiss"))
    faiss.write_index(approx, str(Path(root) / "hnsw.faiss"))


class RecallEngine:
    def __init__(self, root, snapshot):
        import faiss
        root = Path(root)
        self.meta = json.loads((root / "retrieval.json").read_text(encoding="utf-8"))
        check_files(root, self.meta["files"])
        self.snapshot = snapshot
        if self.meta["video_ids"] != snapshot.video_ids:
            raise ValueError("Retrieval index and feature catalog order differ")
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        self.user_index = {u: i + 1 for i, u in enumerate(self.meta["user_ids"])}
        self.model = DualTower(len(self.user_index), len(snapshot.video_ids), self.meta["dim"])
        self.model.load_state_dict(torch.load(root / "dual_tower.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.cf = sparse.load_npz(root / "itemcf.npz")
        self.index = faiss.read_index(str(root / "hnsw.faiss"))
        self.exact = faiss.read_index(str(root / "flat.faiss"))

    def query_vector(self, user):
        with torch.inference_mode():
            return self.model.user_vectors(torch.tensor([self.user_index.get(str(user), 0)])).numpy()

    def routes(self, user, k):
        import faiss
        faiss.omp_set_num_threads(1)
        seen = self.snapshot.seen_indices(user)
        pop = top_indices(self.snapshot.popularity, k, seen)
        positives = self.snapshot.positive_indices(user)
        cf_scores = np.asarray(self.cf[positives].sum(axis=0)).ravel() if len(positives) else np.zeros(len(self.snapshot.video_ids))
        cf = top_indices(cf_scores, k, seen) if len(positives) else pop
        if str(user) not in self.user_index or not len(positives):
            dual = pop
        else:
            # Over-fetch by the number of filtered items, guaranteeing enough
            # survivors even if all previously seen videos are near neighbors.
            limit = min(self.index.ntotal, k + len(seen))
            _, found = self.index.search(self.query_vector(user), limit)
            seen_set = set(seen.tolist())
            dual = np.asarray([i for i in found[0] if i >= 0 and i not in seen_set][:k], dtype=np.int64)
        lists = {"popular": pop, "itemcf": cf, "dual_tower": dual}
        mixed, sources = rrf(lists, k)
        lists["fusion"] = mixed
        return lists, sources

    def recall(self, user, k=300, route="fusion"):
        lists, sources = self.routes(user, k)
        if route not in lists:
            raise ValueError("Unknown recall route")
        ids = lists[route]
        provenance = sources if route == "fusion" else {int(i): [route] for i in ids}
        return ids, provenance

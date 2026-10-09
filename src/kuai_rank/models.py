"""DeepFM and its shared-bottom / task-gated-expert multi-task extensions."""
import torch
from torch import nn
import math


def parameter_budget(sizes, n_numeric, kind, embedding_dim, hidden_dim, experts=4, n_tasks=3):
    tasks = 1 if kind == "deepfm" else n_tasks
    width = len(sizes) * embedding_dim + n_numeric
    count = sum(sizes) * (embedding_dim + tasks) + (n_numeric + 1) * tasks
    bottom = width * hidden_dim + hidden_dim ** 2 + 2 * hidden_dim
    count += bottom * experts if kind == "mmoe" else bottom
    count += tasks * (32 * hidden_dim + 65)
    if kind == "mmoe":
        count += tasks * (width + 1) * experts
    return count


def capacity_matched_hidden(sizes, n_numeric, embedding_dim, hidden_dim, experts=4, n_tasks=3):
    target = parameter_budget(sizes, n_numeric, "mmoe", embedding_dim, hidden_dim, experts, n_tasks)
    width = len(sizes) * embedding_dim + n_numeric
    constant = sum(sizes) * (embedding_dim + 1) + n_numeric + 1 + 65
    linear = width + 34
    root = (-linear + math.sqrt(linear ** 2 + 4 * (target - constant))) / 2
    candidates = [max(1, math.floor(root)), max(1, math.ceil(root))]
    return min(candidates, key=lambda h: abs(parameter_budget(sizes, n_numeric, "deepfm", embedding_dim, h) - target))


def mlp(width, hidden):
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())


class RankModel(nn.Module):
    def __init__(self, sizes, n_numeric, kind="deepfm", embedding_dim=12,
                 hidden_dim=96, experts=4, n_tasks=3):
        super().__init__()
        if kind not in {"deepfm", "sharedbottom", "mmoe"}:
            raise ValueError(f"Unknown model kind {kind}")
        self.kind = kind
        self.n_tasks = 1 if kind == "deepfm" else n_tasks
        self.embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim, padding_idx=0) for size in sizes])
        self.linear_fields = nn.ModuleList([nn.Embedding(size, self.n_tasks, padding_idx=0) for size in sizes])
        self.linear_numeric = nn.Linear(n_numeric, self.n_tasks)
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.01)
            # Unknown values always have a defined, neutral representation.
            with torch.no_grad():
                emb.weight[0].zero_()
        for emb in self.linear_fields:
            nn.init.zeros_(emb.weight)
        width = len(sizes) * embedding_dim + n_numeric
        if kind == "mmoe":
            self.experts = nn.ModuleList([mlp(width, hidden_dim) for _ in range(experts)])
            self.gates = nn.ModuleList([nn.Linear(width, experts) for _ in range(self.n_tasks)])
        else:
            self.bottom = mlp(width, hidden_dim)
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
                                     for _ in range(self.n_tasks)])

    def representation(self, cats, nums):
        emb = torch.stack([layer(cats[:, i]) for i, layer in enumerate(self.embeddings)], dim=1)
        fm = 0.5 * (emb.sum(dim=1).square() - emb.square().sum(dim=1)).sum(dim=1, keepdim=True)
        linear = sum(layer(cats[:, i]) for i, layer in enumerate(self.linear_fields)) + self.linear_numeric(nums)
        x = torch.cat([emb.flatten(start_dim=1), nums], dim=1)
        return x, linear + fm

    def forward(self, cats, nums):
        x, base = self.representation(cats, nums)
        if self.kind == "mmoe":
            experts = torch.stack([expert(x) for expert in self.experts], dim=1)
            reps = [(gate(x).softmax(dim=1).unsqueeze(-1) * experts).sum(dim=1) for gate in self.gates]
        else:
            shared = self.bottom(x)
            reps = [shared] * self.n_tasks
        return torch.cat([tower(rep) for tower, rep in zip(self.towers, reps)], dim=1) + base

    def gate_mean(self, cats, nums):
        if self.kind != "mmoe":
            return None
        x, _ = self.representation(cats, nums)
        return torch.stack([gate(x).softmax(dim=1).mean(dim=0) for gate in self.gates])

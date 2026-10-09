"""Load trusted project artifacts once and share scoring across API and evaluation."""
import json
from pathlib import Path
import pickle

import numpy as np

from .integrity import verify_models


class Scorer:
    def __init__(self, root):
        root = Path(root)
        self.meta = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        verify_models(root, self.meta)
        with (root / "encoder.pkl").open("rb") as f:
            self.encoder = pickle.load(f)
        kind = self.meta["model"]
        if kind == "lightgbm":
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(root / "lightgbm.txt"))
            self.predict_encoded = lambda c, n: model.predict(np.column_stack([c, n]), num_threads=1)[:, None]
        elif kind == "lr":
            from scipy.sparse import hstack, csr_matrix
            with (root / "linear.pkl").open("rb") as f:
                onehot, model = pickle.load(f)
            self.predict_encoded = lambda c, n: model.predict_proba(hstack([onehot.transform(c), csr_matrix(n)], format="csr"))[:, 1:2]
        else:
            import torch
            from .models import RankModel
            from .train import predict_neural
            config = self.meta["config"]
            torch.set_num_threads(config["threads"])
            model = RankModel(self.encoder.sizes, len(self.encoder.numeric), self.meta.get("effective_model_kind", kind),
                              config["embedding_dim"], config["hidden_dim"], config["experts"], len(config["tasks"]))
            model.load_state_dict(torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True))
            model.eval()
            self.predict_encoded = lambda c, n: predict_neural(model, c, n)

    def predict(self, frame):
        cats, nums = self.encoder.transform(frame)
        return self.predict_encoded(cats, nums)

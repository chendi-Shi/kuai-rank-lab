"""Train-only vocabularies and train-only numerical normalization."""
import numpy as np


class FeatureEncoder:
    def __init__(self, categorical, numeric):
        self.categorical = list(categorical)
        self.numeric = list(numeric)

    def fit(self, frame):
        self.vocab = {col: {value: i + 1 for i, value in enumerate(sorted(frame[col].astype(str).unique()))}
                      for col in self.categorical}
        x = frame[self.numeric].to_numpy(dtype=np.float32)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-4)
        return self

    @property
    def sizes(self):
        return [len(self.vocab[col]) + 1 for col in self.categorical]

    def transform(self, frame):
        if len(frame) <= 4096:
            # Series.map(dict) builds a full vocabulary-sized pandas Index on
            # every request. Direct lookups keep small online batches O(rows).
            cats = np.column_stack([np.fromiter((self.vocab[col].get(value, 0)
                                    for value in frame[col].astype(str).to_numpy()),
                                    dtype=np.int64, count=len(frame)) for col in self.categorical])
        else:
            cats = np.column_stack([frame[col].astype(str).map(self.vocab[col]).fillna(0).to_numpy(dtype=np.int64)
                                    for col in self.categorical])
        nums = np.clip((frame[self.numeric].to_numpy(dtype=np.float32) - self.mean) / self.std, -10, 10)
        return cats, nums.astype(np.float32)

    def unknown_rates(self, frame):
        if len(frame) <= 4096:
            return {col: float(np.mean(np.fromiter((value not in self.vocab[col]
                         for value in frame[col].astype(str).to_numpy()), dtype=bool, count=len(frame))))
                    for col in self.categorical}
        return {col: float((~frame[col].astype(str).isin(self.vocab[col])).mean()) for col in self.categorical}

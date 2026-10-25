from __future__ import annotations

import math
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x / denom


class EmbeddingIndex:
    def __init__(self, model_path: str, texts: list[str], names: list[str], device: str = "cuda"):
        self.names = names
        self.texts = texts
        self.model = SentenceTransformer(model_path, device=device)
        self.embeddings = normalize(self.model.encode(texts, batch_size=64, convert_to_numpy=True).astype(np.float32))
        self._tokenized = [_tokenize(text) for text in texts]
        self._doc_freq: dict[str, int] = {}
        for tokens in self._tokenized:
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avg_doc_len = float(np.mean([len(tokens) for tokens in self._tokenized])) if self._tokenized else 1.0

    def query(self, text: str, top_k: int) -> list[dict]:
        vec = normalize(self.model.encode([text], convert_to_numpy=True).astype(np.float32))[0]
        scores = self.embeddings @ vec
        idxs = np.argsort(-scores)[:top_k]
        return [
            {"id": self.names[int(i)], "score": float(scores[int(i)]), "text": self.texts[int(i)]}
            for i in idxs
        ]

    def score_map(self, text: str) -> dict[str, float]:
        vec = normalize(self.model.encode([text], convert_to_numpy=True).astype(np.float32))[0]
        scores = self.embeddings @ vec
        return {self.names[i]: float(scores[i]) for i in range(len(self.names))}

    def bm25_score_map(self, text: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        query_tokens = _tokenize(text)
        if not query_tokens:
            return {name: 0.0 for name in self.names}
        n_docs = max(len(self._tokenized), 1)
        raw_scores = []
        for doc_tokens in self._tokenized:
            doc_len = max(len(doc_tokens), 1)
            tf: dict[str, int] = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                df = self._doc_freq.get(token, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * doc_len / max(self._avg_doc_len, 1e-6))
                score += idf * (freq * (k1 + 1.0)) / denom
            raw_scores.append(score)
        scores_np = np.asarray(raw_scores, dtype=np.float32)
        if float(scores_np.max(initial=0.0)) > 0:
            scores_np = scores_np / max(float(scores_np.max()), 1e-6)
        return {self.names[i]: float(scores_np[i]) for i in range(len(self.names))}

    def hybrid_score_map(self, text: str, dense_weight: float = 0.7, bm25_weight: float = 0.3) -> dict[str, float]:
        dense = self.score_map(text)
        bm25 = self.bm25_score_map(text)
        return {
            name: dense_weight * dense.get(name, 0.0) + bm25_weight * bm25.get(name, 0.0)
            for name in self.names
        }


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("__", ".")
    tokens = re.findall(r"[a-z0-9]+", text)
    split_tokens: list[str] = []
    for token in tokens:
        split_tokens.append(token)
        split_tokens.extend(part for part in token.split("_") if part)
    return [token for token in split_tokens if len(token) > 1]

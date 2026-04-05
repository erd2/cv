"""Сопоставление векторов эмоций: косинусное сходство."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import DatasetItem


@dataclass
class MatchResult:
    item: Optional[DatasetItem]
    similarity: float
    index: int


def best_match(query: np.ndarray, items: List[DatasetItem]) -> MatchResult:
    """Выбор изображения с максимальным косинусным сходством векторов."""
    if not items:
        return MatchResult(item=None, similarity=0.0, index=-1)
    q = query.reshape(1, -1)
    best_idx = 0
    best_sim = -1.0
    for i, it in enumerate(items):
        v = it.vector.reshape(1, -1)
        sim = float(cosine_similarity(q, v)[0, 0])
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return MatchResult(item=items[best_idx], similarity=best_sim, index=best_idx)


def blend_vectors(prev: np.ndarray, nxt: np.ndarray, alpha: float) -> np.ndarray:
    """alpha: 0 = prev, 1 = next."""
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - a) * prev + a * nxt
    s = float(out.sum())
    if s > 1e-8:
        out = out / s
    return out

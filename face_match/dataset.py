"""Загрузка изображений датасета, метаданные и кеш эмбеддингов."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from . import config


@dataclass
class DatasetItem:
    path: Path
    label: Optional[str]  # из metadata.json, если задано
    vector: np.ndarray = field(repr=False)


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class DatasetCache:
    """Кеш векторов эмоций для изображений датасета."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: Dict[str, Any] = {}
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self._data = pickle.load(f)

    def _key_for_file(self, image_path: Path) -> str:
        return hashlib.sha256(_file_signature(image_path).encode()).hexdigest()[:16]

    def get_vector(self, image_path: Path) -> Optional[np.ndarray]:
        key = self._key_for_file(image_path)
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.get("sig") != _file_signature(image_path):
            return None
        return np.array(entry["vector"], dtype=np.float64)

    def set_vector(self, image_path: Path, vector: np.ndarray) -> None:
        key = self._key_for_file(image_path)
        self._data[key] = {
            "sig": _file_signature(image_path),
            "vector": vector.tolist(),
            "path": str(image_path),
        }

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._data, f)

    def clear(self) -> None:
        self._data = {}
        if self.cache_path.exists():
            self.cache_path.unlink()


def list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not folder.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def load_image_bgr(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # RGB -> BGR для OpenCV / FER
    return arr[:, :, ::-1].copy()

"""Построение / обновление кеша векторов для всех изображений в data/images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .dataset import DatasetCache, list_image_files, load_image_bgr, load_metadata
from .detector import MediaPipeFaceDetector
from .emotion import EmotionRecognizer


def _label_to_vector(label: str) -> np.ndarray:
    label = label.strip().lower()
    if label not in config.EMOTION_KEYS:
        raise ValueError(f"Неизвестная эмоция в metadata: {label!r}. Допустимо: {config.EMOTION_KEYS}")
    vec = np.zeros(len(config.EMOTION_KEYS), dtype=np.float64)
    vec[config.EMOTION_KEYS.index(label)] = 1.0
    return vec


def build_vectors(
    images_dir: Path,
    cache_path: Path,
    metadata_path: Path,
    rebuild: bool,
) -> None:
    meta = load_metadata(metadata_path)
    labels_by_name = meta.get("labels", {}) if isinstance(meta.get("labels"), dict) else {}

    cache = DatasetCache(cache_path)
    if rebuild:
        cache.clear()
        cache = DatasetCache(cache_path)

    paths = list_image_files(images_dir)
    if not paths:
        print(f"Нет изображений в {images_dir}. Добавьте файлы (.jpg, .png, …).")
        return

    detector = MediaPipeFaceDetector()
    emotion = EmotionRecognizer()

    try:
        for i, path in enumerate(paths):
            print(f"[{i + 1}/{len(paths)}] {path.name}")
            cached = None if rebuild else cache.get_vector(path)
            if cached is not None:
                continue

            forced = labels_by_name.get(path.name)
            if forced:
                try:
                    vec = _label_to_vector(str(forced))
                except ValueError as e:
                    print(e)
                    continue
                cache.set_vector(path, vec)
                continue

            try:
                bgr = load_image_bgr(path)
            except OSError as e:
                print(f"Пропуск {path}: {e}")
                continue

            h, w = bgr.shape[:2]
            if max(h, w) > 960:
                scale = 960 / max(h, w)
                import cv2

                bgr = cv2.resize(
                    bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            face = detector.largest(bgr)
            if face is not None:
                vec, _ = emotion.predict(bgr, face)
            else:
                vec, _ = emotion.predict_full_image(bgr)

            cache.set_vector(path, vec)
    finally:
        detector.close()

    cache.save()
    print(f"Готово. Кеш: {cache_path} ({len(paths)} файлов).")


def main() -> None:
    p = argparse.ArgumentParser(description="Построить кеш эмбеддингов датасета")
    p.add_argument("--rebuild", action="store_true", help="Пересчитать все векторы")
    args = p.parse_args()

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    build_vectors(
        config.IMAGES_DIR,
        config.CACHE_DIR / "embeddings.pkl",
        config.METADATA_FILE,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()

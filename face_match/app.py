"""Главный цикл: веб-камера, детекция, эмоции, сопоставление, отображение."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .dataset import DatasetCache, DatasetItem, list_image_files, load_image_bgr
from .detector import FaceBox, MediaPipeFaceDetector
from .emotion import EmotionRecognizer
from .emotion_logger import log_emotion, setup_emotion_logger
from .matcher import MatchResult, best_match


def _resize_frame_for_fer(frame_bgr: np.ndarray, face: FaceBox, max_w: int = 360) -> Tuple[np.ndarray, FaceBox]:
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr, face
    scale = max_w / w
    nw, nh = int(w * scale), int(h * scale)
    small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sf = FaceBox(
        int(face.x * scale),
        int(face.y * scale),
        int(face.w * scale),
        int(face.h * scale),
    ).clip(nw, nh)
    return small, sf


def _load_match_image(path: Path, panel_h: int) -> np.ndarray:
    bgr = load_image_bgr(path)
    ih, iw = bgr.shape[:2]
    scale = panel_h / ih
    new_w = max(1, int(iw * scale))
    return cv2.resize(bgr, (new_w, panel_h), interpolation=cv2.INTER_AREA)


def _play_match_sound() -> None:
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        pass


class SmoothMatcherDisplay:
    """Плавная смена превью совпадения."""

    def __init__(self, panel_h: int, blend_frames: int) -> None:
        self.panel_h = panel_h
        self.blend_frames = max(1, blend_frames)
        self._prev_img: Optional[np.ndarray] = None
        self._next_img: Optional[np.ndarray] = None
        self._frame_i = 0
        self._last_path: Optional[Path] = None

    def update(self, path: Path | None, cache_loader) -> np.ndarray:
        """cache_loader: Path -> BGR image resized to panel."""
        if path is None:
            blank = np.zeros((self.panel_h, self.panel_h, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "No match",
                (20, self.panel_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            self._prev_img = blank
            self._next_img = None
            return blank

        if self._last_path != path:
            new_bgr = cache_loader(path)
            if self._prev_img is None:
                self._prev_img = new_bgr
            else:
                self._next_img = new_bgr
                self._frame_i = 0
            self._last_path = path

        if self._next_img is None:
            assert self._prev_img is not None
            return self._prev_img

        alpha = min(1.0, (self._frame_i + 1) / self.blend_frames)
        w = max(self._prev_img.shape[1], self._next_img.shape[1])
        h = self.panel_h
        a = cv2.resize(self._prev_img, (w, h))
        b = cv2.resize(self._next_img, (w, h))
        out = cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)
        self._frame_i += 1
        if self._frame_i >= self.blend_frames:
            self._prev_img = self._next_img
            self._next_img = None
        return out


def load_dataset_items(images_dir: Path, cache: DatasetCache) -> List[DatasetItem]:
    paths = list_image_files(images_dir)
    items: List[DatasetItem] = []
    for p in paths:
        vec = cache.get_vector(p)
        if vec is None:
            print(f"Пропуск (нет в кеше): {p.name} — запустите: python -m face_match.index_dataset")
            continue
        items.append(DatasetItem(path=p, label=None, vector=vec))
    return items


def run(
    camera_index: int = 0,
    sound: bool = True,
    log_interval_s: float = 1.0,
) -> None:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = config.CACHE_DIR / "embeddings.pkl"
    cache = DatasetCache(cache_path)
    items = load_dataset_items(config.IMAGES_DIR, cache)
    if not items:
        print("Датасет пуст или кеш не построен. Положите изображения в data/images и выполните:")
        print("  python -m face_match.index_dataset")
        return

    detector = MediaPipeFaceDetector()
    emotion = EmotionRecognizer()
    logger = setup_emotion_logger(config.LOG_FILE)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    panel_h = config.FRAME_HEIGHT
    match_preview_cache: dict = {}

    def get_panel(path: Path) -> np.ndarray:
        if path not in match_preview_cache:
            match_preview_cache[path] = _load_match_image(path, panel_h)
        return match_preview_cache[path]

    smooth = SmoothMatcherDisplay(panel_h, config.MATCH_BLEND_FRAMES)

    last_emo_time = 0.0
    last_log_time = 0.0
    query_vec = np.ones(len(config.EMOTION_KEYS), dtype=np.float64) / len(config.EMOTION_KEYS)
    last_match: MatchResult = best_match(query_vec, items)
    last_match_idx = last_match.index

    window = "Face Match"
    print("Окно:", window, "| Q — выход | S — звук вкл/выкл")
    sound_on = sound

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            faces = detector.detect(frame)
            for fb in faces:
                cv2.rectangle(
                    frame,
                    (fb.x, fb.y),
                    (fb.x + fb.w, fb.y + fb.h),
                    (0, 255, 0),
                    2,
                )
            if len(faces) > 1:
                cv2.putText(
                    frame,
                    f"faces: {len(faces)} (match: largest)",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 255),
                    1,
                )

            now = time.perf_counter()
            if faces and (now - last_emo_time) * 1000 >= config.MIN_PROCESS_INTERVAL_MS:
                last_emo_time = now
                primary = faces[0]
                small, sface = _resize_frame_for_fer(frame, primary)
                query_vec, _ = emotion.predict(small, sface)
                last_match = best_match(query_vec, items)

                if sound_on and last_match.index >= 0:
                    if (
                        last_match.index != last_match_idx
                        and last_match.similarity >= config.SIMILARITY_THRESHOLD_SOUND
                    ):
                        _play_match_sound()
                    last_match_idx = last_match.index

                dom = emotion.dominant_label(query_vec)
                if now - last_log_time >= log_interval_s:
                    last_log_time = now
                    mname = (
                        last_match.item.path.name
                        if last_match.item is not None
                        else "—"
                    )
                    log_emotion(logger, dom, last_match.similarity, mname)

            dom_label = emotion.dominant_label(query_vec)
            cv2.putText(
                frame,
                f"Emotion: {dom_label}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
            )
            if last_match.item is not None:
                cv2.putText(
                    frame,
                    f"match: {last_match.similarity:.2f}",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 255, 180),
                    2,
                )

            path = last_match.item.path if last_match.item else None
            panel = smooth.update(path, get_panel)

            gap = 8
            total_w = frame.shape[1] + gap + panel.shape[1]
            canvas = np.zeros((panel_h, total_w, 3), dtype=np.uint8)
            canvas[:, : frame.shape[1]] = frame
            canvas[:, frame.shape[1] + gap :] = panel

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                sound_on = not sound_on
                print("Звук:", "on" if sound_on else "off")

            elapsed = time.perf_counter() - t0
            target = 1.0 / config.TARGET_FPS
            if elapsed < target:
                time.sleep(target - elapsed)
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Face Match — камера и сопоставление эмоций")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Индекс веб-камеры")
    ap.add_argument("--no-sound", action="store_true", help="Отключить звук при совпадении")
    ap.add_argument("--log-interval", type=float, default=1.0, help="Интервал записи в лог, сек")
    args = ap.parse_args()
    run(camera_index=args.camera, sound=not args.no_sound, log_interval_s=args.log_interval)


if __name__ == "__main__":
    main()

"""Детекция лиц через MediaPipe Face Detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from . import config


@dataclass
class FaceBox:
    """Прямоугольник лица в пикселях: x, y от левого верхнего угла, ширина и высота."""

    x: int
    y: int
    w: int
    h: int

    def clip(self, frame_w: int, frame_h: int) -> "FaceBox":
        x = max(0, self.x)
        y = max(0, self.y)
        w = min(self.w, frame_w - x)
        h = min(self.h, frame_h - y)
        return FaceBox(x, y, max(1, w), max(1, h))

    def area(self) -> int:
        return self.w * self.h


class MediaPipeFaceDetector:
    def __init__(self) -> None:
        self._mp_fd = mp.solutions.face_detection
        self._detector = self._mp_fd.FaceDetection(
            model_selection=config.MP_MODEL_SELECTION,
            min_detection_confidence=config.MP_MIN_DET_CONF,
        )

    def close(self) -> None:
        self._detector.close()

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        """Возвращает список лиц, отсортированный по убыванию площади."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._detector.process(rgb)
        if not res.detections:
            return []
        boxes: List[FaceBox] = []
        for det in res.detections:
            rb = det.location_data.relative_bounding_box
            x = int(rb.xmin * w)
            y = int(rb.ymin * h)
            bw = int(rb.width * w)
            bh = int(rb.height * h)
            box = FaceBox(x, y, bw, bh).clip(w, h)
            boxes.append(box)
        boxes.sort(key=lambda b: b.area(), reverse=True)
        return boxes

    def largest(self, frame_bgr: np.ndarray) -> FaceBox | None:
        faces = self.detect(frame_bgr)
        return faces[0] if faces else None

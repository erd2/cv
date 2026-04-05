"""Распознавание эмоций: ONNX FER+ (8 классов) → вектор из 7 меток как в классическом FER."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from . import config
from .detector import FaceBox


MODEL_FILENAME = "emotion-ferplus-8.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/"
    + MODEL_FILENAME
)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / np.sum(e)).astype(np.float64)


def _map_ferplus8_to_fer7(p8: np.ndarray) -> np.ndarray:
    """
    FER+ порядок: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
    FER7: angry, disgust, fear, happy, neutral, sad, surprise
    """
    neutral, happiness, surprise, sadness, anger, disgust, fear, contempt = (
        float(p8[0]),
        float(p8[1]),
        float(p8[2]),
        float(p8[3]),
        float(p8[4]),
        float(p8[5]),
        float(p8[6]),
        float(p8[7]),
    )
    # презрение частично в disgust и neutral
    vec = np.array(
        [
            anger,
            disgust + 0.5 * contempt,
            fear,
            happiness,
            neutral + 0.5 * contempt,
            sadness,
            surprise,
        ],
        dtype=np.float64,
    )
    s = float(vec.sum())
    if s > 1e-12:
        vec = vec / s
    else:
        vec = np.ones(7, dtype=np.float64) / 7.0
    return vec


def _ensure_model(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка модели эмоций (~34 МБ): {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Сохранено: {path}")


def _crop_bgr(frame_bgr: np.ndarray, face: FaceBox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, face.x)
    y1 = max(0, face.y)
    x2 = min(w, face.x + face.w)
    y2 = min(h, face.y + face.h)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]


def _to_input_1x1x64x64(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    return x.reshape(1, 1, 64, 64)


class EmotionRecognizer:
    """ONNX FER+: вектор вероятностей по порядку config.EMOTION_KEYS (7 эмоций)."""

    def __init__(self, model_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parent.parent
        self._model_path = model_path or (root / "models" / MODEL_FILENAME)
        _ensure_model(self._model_path)
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def vector_from_dict(self, emotions: Dict[str, float]) -> np.ndarray:
        return np.array([float(emotions.get(k, 0.0)) for k in config.EMOTION_KEYS], dtype=np.float64)

    def dominant_label(self, vec: np.ndarray) -> str:
        idx = int(np.argmax(vec))
        return config.EMOTION_KEYS[idx]

    def _infer_face_bgr(self, face_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if face_bgr.size == 0:
            uniform = np.ones(len(config.EMOTION_KEYS), dtype=np.float64) / len(config.EMOTION_KEYS)
            return uniform, {k: float(uniform[i]) for i, k in enumerate(config.EMOTION_KEYS)}
        inp = _to_input_1x1x64x64(face_bgr)
        out = self._session.run(None, {self._input_name: inp})[0]
        scores = np.asarray(out).reshape(-1)
        p8 = _softmax(scores)
        vec = _map_ferplus8_to_fer7(p8)
        emo = {k: float(vec[i]) for i, k in enumerate(config.EMOTION_KEYS)}
        return vec, emo

    def predict(
        self,
        frame_bgr: np.ndarray,
        face: FaceBox,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        crop = _crop_bgr(frame_bgr, face)
        return self._infer_face_bgr(crop)

    def predict_full_image(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Вся сцена: сначала Haar-лицо, иначе весь кадр, приведённый к 64×64."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) > 0:
            areas = [w * h for (_, _, w, h) in faces]
            i = int(np.argmax(areas))
            x, y, w, h = faces[i]
            crop = frame_bgr[y : y + h, x : x + w]
            return self._infer_face_bgr(crop)
        return self._infer_face_bgr(frame_bgr)

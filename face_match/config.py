"""Параметры приложения по умолчанию."""

from pathlib import Path

# Корень проекта (родитель каталога face_match)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CACHE_DIR = DATA_DIR / "cache"
METADATA_FILE = DATA_DIR / "metadata.json"
LOG_DIR = PROJECT_ROOT / "logs"

# Видео
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 24
MIN_PROCESS_INTERVAL_MS = 1000 // 25  # не чаще ~25 раз/с для тяжёлой модели

# Детекция (MediaPipe Short-range, быстрее для веб-камеры)
MP_MIN_DET_CONF = 0.5
MP_MODEL_SELECTION = 0  # 0 = short range (2m), 1 = full range

# Сопоставление
EMOTION_KEYS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
MATCH_BLEND_FRAMES = 12
SIMILARITY_THRESHOLD_SOUND = 0.72  # проиграть звук при сильном совпадении

# Логи
LOG_FILE = LOG_DIR / "emotions.log"

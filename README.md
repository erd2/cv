# Face Match

Приложение на Python: веб-камера в реальном времени, детекция лица (MediaPipe), оценка эмоций (ONNX FER+) и сопоставление с изображениями из локального датасета по косинусному сходству векторов эмоций.

## Требования

- Python 3.10+ (проверено с 3.14)
- Веб-камера
- Доступ в интернет при первом запуске (скачивание ONNX-модели, около 34 МБ, в каталог `models/`)

## Установка

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\pip install -r requirements.txt
```

Linux / macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Подготовка датасета

1. Поместите изображения (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) в `data/images/`.
2. При необходимости задайте ручные метки эмоций в `data/metadata.json`:

```json
{
  "labels": {
    "example.jpg": "happy"
  }
}
```

Допустимые значения: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. Для таких файлов вектор в кеше строится как one-hot по метке, без запуска модели на изображении.

3. Постройте кеш эмбеддингов:

```bash
python -m face_match.index_dataset
```

Полный пересчёт всех векторов:

```bash
python -m face_match.index_dataset --rebuild
```

Кеш хранится в `data/cache/embeddings.pkl`.

## Запуск

```bash
python -m face_match
```

Параметры:

- `--camera N` — индекс камеры (по умолчанию `0`)
- `--no-sound` — отключить звук при сильном совпадении и смене картинки
- `--log-interval SEC` — минимальный интервал записи строки в лог (секунды, по умолчанию `1.0`)

В окне: **Q** или **Esc** — выход; **S** — переключить звук.

## Поведение

- Отображается видео с камеры и панель с наиболее похожим изображением из датасета.
- Несколько лиц: отрисовываются все обнаруженные лица; для сопоставления используется самое крупное.
- Плавная смена превью при смене лучшего совпадения.
- Логи эмоций и совпадений: `logs/emotions.log`.

Частота кадров отображения настраивается в `face_match/config.py` (`TARGET_FPS`, `MIN_PROCESS_INTERVAL_MS`).

## Структура проекта

| Путь | Описание |
|------|----------|
| `face_match/app.py` | Главный цикл и интерфейс OpenCV |
| `face_match/detector.py` | Детекция лиц (MediaPipe) |
| `face_match/emotion.py` | ONNX FER+, маппинг в 7 эмоций |
| `face_match/dataset.py` | Список файлов и кеш векторов |
| `face_match/matcher.py` | Косинусное сходство |
| `face_match/index_dataset.py` | CLI индексации датасета |
| `face_match/config.py` | Пути и параметры |
| `data/images/` | Изображения датасета |
| `data/metadata.json` | Опциональные метки эмоций |
| `data/cache/` | Файл кеша эмбеддингов |
| `models/` | ONNX-модель (после первой загрузки) |

## Стек

OpenCV, MediaPipe, NumPy, Pillow, scikit-learn, ONNX Runtime.

## Лицензия модели

ONNX FER+ взята из [ONNX Model Zoo](https://github.com/onnx/models) (MIT). Подробности в описании модели `emotion_ferplus`.

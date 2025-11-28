# Детекция людей и поездов

Система детекции объектов (люди и поезда) на изображениях и видео с использованием YOLOv11.
Оптимизировано для работы на CPU с настройками производительности.

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Модель YOLO11n.pt будет автоматически скачана при первом запуске (~7 MB).

## Использование

1. Откройте `config.json` и укажите путь к вашему файлу:
```json
{
  "input_file": "video.mov"
}
```

2. Запустите:
```bash
python run.py
```
или
```bash
python src/main.py
```

3. Откроется окно с результатом детекции. Нажмите 'q' для выхода (видео) или любую клавишу (изображение).

## Структура проекта

Проект организован в модульную структуру:

```
xakatonAI/
├── src/
│   ├── __init__.py          # Пакет модулей
│   ├── main.py              # Точка входа приложения
│   ├── detector.py          # Класс YOLODetector для детекции объектов
│   ├── config_loader.py    # Загрузка и валидация config.json
│   ├── image_utils.py       # Утилиты для работы с изображениями (чтение, ресайз)
│   ├── filters.py           # Цветовые фильтры для детекций
│   └── processors.py        # Обработка изображений и видео
├── config.json              # Конфигурационный файл
├── requirements.txt        # Зависимости проекта
├── run.py                  # Обёртка для запуска (рекомендуется)
└── README.md               # Документация
```

### Описание модулей

- **`detector.py`** — класс `YOLODetector` для работы с моделью YOLO11, детекция объектов, отрисовка результатов
- **`config_loader.py`** — загрузка и валидация конфигурации из JSON
- **`image_utils.py`** — функции для чтения изображений с поддержкой кириллицы, изменения размера кадров, извлечения ROI
- **`filters.py`** — цветовые фильтры: проверка детекций по RGB/HSV диапазонам, anti-color фильтры
- **`processors.py`** — функции `process_image()` и `process_video()` для обработки входных файлов
- **`main.py`** — главная функция, инициализация детектора, выбор типа обработки

## Настройки оптимизации

В `config.json` можно настроить производительность:

### Видео оптимизация (`video_optimization`):
- `target_fps` - целевой FPS обработки (по умолчанию 10)
- `max_width` - максимальная ширина кадра (по умолчанию 1280)
- `max_height` - максимальная высота кадра (по умолчанию 720)
- `frame_skip` - пропуск кадров (по умолчанию 1, обрабатываются все)
- `maintain_aspect_ratio` - сохранять пропорции (true/false)

### Детекция (`detection`):
- `confidence_threshold` - порог уверенности (0.0 - 1.0), по умолчанию 0.5

### Цвета (`colors`):
- Позволяет настроить цвет рамок для каждого класса
- Ключ можно задать по имени (`"person"`) или по ID (`"0"`)
- Значение указывается в формате BGR (Blue, Green, Red), например `[255, 0, 0]`

### Цветовые фильтры (`color_filters`):
- `enabled` — включает/отключает фильтрацию (true/false)
- Для каждого класса можно указать:
  - `min_rgb` / `max_rgb` — диапазон цветов в BGR
  - `min_hsv` / `max_hsv` — диапазон цветов в HSV
  - `match_threshold` — доля пикселей, которые должны попасть в положительный диапазон (0..1)
  - `anti_color_hsv` + `anti_color_range` — центр оттенка, который нужно исключить, и радиус (в HSV)
  - `anti_match_threshold` — доля “запрещённого” цвета, при которой детект отклоняется
- Можно задавать классы по имени (`"train"`) или ID (`"6"`)
- При включённом `debug.log_detection_details` в консоль выводятся подробности прохождения фильтра


## Пример конфига

```json
{
  "input_file": "video.mov",
  "yolo": {
    "model": "yolo11n.pt",
    "device": "cpu"
  },
  "detection": {
    "confidence_threshold": 0.5,
    "target_classes": [0, 6]
  },
  "colors": {
    "person": [0, 255, 0],
    "train": [255, 0, 0],
    "6": [150, 0, 255]
  },
  "color_filters": {
    "enabled": true,
    "train": {
      "anti_color_hsv": [90, 50, 30],
      "anti_color_range": 30,
      "anti_match_threshold": 0.2
    }
  },
  "processing": {
    "show_preview": true,
    "save_results": true,
    "output_dir": "results",
    "half_precision": false
  },
  "debug": {
    "show_filtered_objects": true,
    "log_detection_details": true
  },
  "video_optimization": {
    "target_fps": 10,
    "max_width": 1280,
    "max_height": 720,
    "keep_width_native": true,
    "maintain_aspect_ratio": true
  }
}
```

# Детекция людей и поездов (YOLO11) - Modular Version

Система детекции объектов (люди и поезда) на изображениях и видео с использованием YOLO11. Включает трекинг объектов, распознавание номеров поездов с помощью OCR и цветовую фильтрацию детекций.

## Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Описание модулей](#описание-модулей)
- [Конфигурация](#конфигурация)
- [Примеры использования](#примеры-использования)

## Установка

### Требования

- Python 3.8+
- CUDA (опционально, для GPU ускорения)

### Шаги установки

1. Клонируйте репозиторий или распакуйте архив проекта

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Модель YOLO11 будет автоматически скачана при первом запуске:
   - `yolo11n.pt` - nano (самая быстрая, ~7 MB)
   - `yolo11m.pt` - medium (баланс скорости/точности, ~50 MB)
   - `yolo11l.pt` - large (высокая точность, ~100 MB)

## Быстрый старт

1. Откройте `config.json` и укажите путь к вашему файлу:
```json
{
  "input_file": "people_and_work.mp4"
}
```

2. Запустите обработку:
```bash
python main.py
```

3. Для видео: нажмите `q` для выхода
   Для изображений: нажмите любую клавишу для закрытия окна

## Описание модулей

### `main.py`
Главный модуль приложения. Выполняет:
- Загрузку конфигурации из `config.json`
- Инициализацию детектора YOLO11
- Определение типа входного файла (изображение/видео)
- Запуск соответствующей функции обработки

**Основные функции:**
- `main()` - главная функция приложения

### `src/config_loader.py`
Модуль для загрузки и валидации конфигурации.

**Функции:**
- `load_config(config_path="config.json")` - загружает JSON конфигурацию, проверяет существование файла и корректность формата

### `src/detector.py`
Модуль детекции объектов на основе YOLO11.

**Класс `YOLODetector`:**
- `__init__(model_path, conf_threshold, device, custom_colors, half_precision)` - инициализация детектора
- `detect(frame, target_classes)` - детекция объектов на кадре, возвращает список `[(class_id, confidence, x1, y1, x2, y2), ...]`
- `draw_detections(frame, detections, show_track_ids, train_numbers)` - отрисовка детекций на кадре

**Поддерживаемые классы:**
- `0` - person (человек)
- `6` - train (поезд)

### `src/processors.py`
Модуль обработки изображений и видео.

**Функции:**
- `process_image(image_path, detector, config)` - обработка одного изображения
  - Загрузка изображения
  - Оптимизация размера (если нужно)
  - Детекция объектов
  - Применение цветовых фильтров
  - Отрисовка результатов
  - Сохранение и показ результата

- `process_video(video_path, detector, config)` - обработка видео
  - Чтение видео с поддержкой кириллицы в пути
  - Оптимизация FPS и размера кадров
  - Инициализация трекера (re-identification)
  - Инициализация OCR для распознавания номеров
  - Обработка каждого кадра
  - Трекинг объектов с присвоением ID
  - Распознавание номеров поездов
  - Сохранение результата в видео файл
  - Показ превью в реальном времени

### `src/filters.py`
Модуль цветовой фильтрации детекций.

**Функции:**
- `apply_color_filters(frame, detections, filters_cfg, class_map, debug_cfg)` - применяет цветовые фильтры к детекциям
  - Возвращает отфильтрованные детекции и список отклоненных
  - Поддерживает фильтрацию по RGB и HSV диапазонам
  - Поддерживает anti-color фильтры (исключение определенных цветов)

- `passes_color_filter(roi, cfg)` - проверяет, проходит ли ROI цветовой фильтр
  - Проверка по min_rgb/max_rgb
  - Проверка по min_hsv/max_hsv
  - Проверка anti_color (исключение цветов)

- `annotate_rejected(frame, rejected, detector, color)` - отрисовывает отклоненные объекты на кадре

### `src/image_utils.py`
Утилиты для работы с изображениями.

**Функции:**
- `imread_unicode(image_path)` - чтение изображения с поддержкой кириллицы в пути
- `resize_frame(frame, max_width, max_height, maintain_aspect, keep_width_native)` - изменение размера кадра с сохранением пропорций
- `extract_roi(frame, x1, y1, x2, y2)` - извлечение области интереса (ROI) из кадра

### `src/tracker.py`
Модуль трекинга объектов с использованием re-identification.

**Класс `ReIDTracker`:**
- `__init__(feature_extractor, max_distance, max_age, min_hits, iou_threshold)` - инициализация трекера
- `update(frame, detections)` - обновление трекера новыми детекциями, возвращает список треков с ID: `[(track_id, class_id, confidence, x1, y1, x2, y2), ...]`

**Класс `Track`:**
- Хранит информацию о треке: позицию, признаки, возраст, ID класса, номер поезда

**Особенности:**
- Использует извлечение признаков для сопоставления объектов между кадрами
- Kalman фильтр для предсказания позиции
- IoU и cosine distance для сопоставления

### `src/reid.py`
Модуль извлечения признаков для re-identification.

**Класс `FeatureExtractor`:**
- `__init__(device, feature_dim)` - инициализация экстрактора признаков
- `extract_features(frame, detections)` - извлечение признаков для всех детекций
- `compute_distance(features1, features2)` - вычисление расстояния между признаками (cosine distance)
- `compute_distance_matrix(features1, features2)` - матрица расстояний между наборами признаков

**Архитектура модели:**
- Упрощенная CNN на основе ResNet
- Выходной размер признаков: 128 (по умолчанию)

### `src/ocr_reader.py`
Модуль распознавания номеров поездов с помощью OCR.

**Класс `TrainNumberOCR`:**
- `__init__(ocr_engine, languages)` - инициализация OCR (EasyOCR или Tesseract)
- `preprocess_image(roi)` - предобработка изображения для улучшения распознавания:
  - Конвертация в grayscale
  - Улучшение контраста (CLAHE)
  - Адаптивная бинаризация
  - Морфологические операции
  - Увеличение размера для лучшего распознавания
- `recognize_text_easyocr(roi)` - распознавание с помощью EasyOCR
- `recognize_text_tesseract(roi)` - распознавание с помощью Tesseract
- `recognize_train_number(frame, roi_config)` - распознавание из заданной области
- `recognize_from_right_half(frame)` - распознавание из правой половины кадра
- `recognize_from_train_bbox(frame, bbox, roi_offset)` - распознавание из области детектированного поезда

**Поддерживаемые OCR движки:**
- EasyOCR (рекомендуется) - поддерживает русский и английский языки
- Tesseract - альтернативный вариант

## Конфигурация

Все настройки находятся в файле `config.json`. Ниже описаны все доступные параметры.

### Основные настройки

#### `input_file`
Путь к входному файлу (изображение или видео).
```json
"input_file": "people_and_work.mp4"
```

### Настройки YOLO (`yolo`)

#### `model`
Путь к модели YOLO11 или имя модели для автоматической загрузки.
- `yolo11n.pt` - nano (самая быстрая)
- `yolo11m.pt` - medium (рекомендуется)
- `yolo11l.pt` - large (высокая точность)
```json
"model": "yolo11m.pt"
```

#### `device`
Устройство для обработки. Должно быть `"cpu"` или `"cuda"` (для GPU). Если указано `"gpu"`, будет автоматически преобразовано в `"cuda"`.
```json
"device": "cpu"
```

### Настройки детекции (`detection`)

#### `confidence_threshold`
Порог уверенности детекции (0.0 - 1.0). Объекты с уверенностью ниже этого значения будут проигнорированы.
```json
"confidence_threshold": 0.5
```

#### `target_classes`
Список ID классов для детекции. `[0]` - только люди, `[6]` - только поезда, `[0, 6]` - оба класса.
```json
"target_classes": [0, 6]
```

#### `class_names`
Словарь соответствия ID классов и их имен (опционально, для отображения).
```json
"class_names": {
  "0": "person",
  "6": "train"
}
```

### Настройки цветов (`colors`)

Цвета рамок для каждого класса в формате BGR `[Blue, Green, Red]`. Ключ может быть именем класса или ID.
```json
"colors": {
  "person": [0, 255, 0],    // Зеленый для людей
  "train": [255, 0, 0],     // Синий для поездов
  "6": [150, 0, 255]        // Альтернативный способ указания по ID
}
```

### Цветовые фильтры (`color_filters`)

Фильтрация детекций по цвету для уменьшения ложных срабатываний.

#### `enabled`
Включить/выключить цветовую фильтрацию.
```json
"enabled": true
```

#### Настройки для каждого класса

Для каждого класса можно указать фильтры. Ключ может быть именем (`"train"`) или ID (`"6"`).

**Положительные фильтры (объект должен содержать цвет):**
- `min_rgb` / `max_rgb` - диапазон цветов в BGR формате `[B, G, R]`
- `min_hsv` / `max_hsv` - диапазон цветов в HSV формате `[H, S, V]`
- `match_threshold` - доля пикселей (0.0-1.0), которые должны попасть в диапазон

**Отрицательные фильтры (anti-color, исключение цветов):**
- `anti_color_hsv` - центр оттенка в HSV, который нужно исключить `[H, S, V]`
- `anti_color_rgb` - центр оттенка в RGB, который нужно исключить `[B, G, R]`
- `anti_color_range` - радиус диапазона для исключения (целое число)
- `anti_match_threshold` - доля пикселей (0.0-1.0) "запрещенного" цвета, при которой детекция отклоняется

**Пример:**
```json
"color_filters": {
  "enabled": true,
  "train": {
    "anti_color_hsv": [90, 50, 30],      // Исключить зеленоватые оттенки
    "anti_color_range": 30,               // Радиус исключения
    "anti_match_threshold": 0.4           // Если 40% пикселей зеленые - отклонить
  }
}
```

### Оптимизация видео (`video_optimization`)

#### `target_fps`
Целевой FPS обработки. Если исходное видео имеет больший FPS, кадры будут пропускаться автоматически.
```json
"target_fps": 20
```

#### `max_width` / `max_height`
Максимальные размеры кадра. Если кадр больше, он будет уменьшен с сохранением пропорций.
```json
"max_width": 1280,
"max_height": 540
```

#### `frame_skip`
Явный пропуск кадров. Если `1` - обрабатываются все кадры. Если больше `1`, используется явный пропуск вместо автоматического расчета по `target_fps`.
```json
"frame_skip": 1
```

#### `maintain_aspect_ratio`
Сохранять пропорции при изменении размера.
```json
"maintain_aspect_ratio": true
```

#### `keep_width_native`
Сохранять исходную ширину, изменять только высоту (если превышает `max_height`).
```json
"keep_width_native": true
```

### Настройки обработки (`processing`)

#### `show_preview`
Показывать окно с превью во время обработки.
```json
"show_preview": true
```

#### `half_precision`
Использовать половинную точность (FP16) для ускорения на GPU. Не рекомендуется для CPU.
```json
"half_precision": false
```

#### `save_results`
Сохранять результаты обработки в файлы.
```json
"save_results": true
```

#### `output_dir`
Папка для сохранения результатов.
```json
"output_dir": "results"
```

### Настройки отладки (`debug`)

#### `show_filtered_objects`
Показывать отклоненные цветовым фильтром объекты красным цветом.
```json
"show_filtered_objects": true
```

#### `log_detection_details`
Выводить в консоль подробную информацию о прохождении цветовых фильтров.
```json
"log_detection_details": true
```

### Re-identification трекинг (`re_identification`)

#### `enabled`
Включить/выключить трекинг объектов с присвоением ID.
```json
"enabled": true
```

#### `max_distance`
Максимальное расстояние признаков (0.0-1.0) для сопоставления объектов между кадрами.
```json
"max_distance": 0.5
```

#### `max_age`
Максимальный возраст трека без обновления (в кадрах). Трек удаляется, если не обновлялся столько кадров.
```json
"max_age": 10
```

#### `min_hits`
Минимальное количество попаданий для подтверждения трека. Треки с меньшим количеством считаются временными.
```json
"min_hits": 8
```

#### `iou_threshold`
Порог IoU (Intersection over Union) для сопоставления по позиции (0.0-1.0).
```json
"iou_threshold": 0.3
```

### OCR распознавание номеров (`train_number_ocr`)

#### `enabled`
Включить/выключить распознавание номеров поездов.
```json
"enabled": true
```

#### `engine`
Движок OCR: `"easyocr"` (рекомендуется) или `"tesseract"`.
```json
"engine": "easyocr"
```

#### `frame_skip`
Проверять номер поезда только каждый N-й кадр (для оптимизации производительности).
```json
"frame_skip": 60
```

#### `use_bottom_right_quadrant`
Использовать правую половину кадра для поиска номера поезда (оптимизированный метод).
```json
"use_bottom_right_quadrant": true
```

## Примеры использования

### Пример 1: Базовая детекция на изображении

```json
{
  "input_file": "test_image.jpg",
  "yolo": {
    "model": "yolo11n.pt",
    "device": "cpu"
  },
  "detection": {
    "confidence_threshold": 0.5,
    "target_classes": [0, 6]
  },
  "processing": {
    "show_preview": true,
    "save_results": true,
    "output_dir": "results"
  }
}
```

### Пример 2: Обработка видео с трекингом и OCR

```json
{
  "input_file": "people_and_work.mp4",
  "yolo": {
    "model": "yolo11m.pt",
    "device": "cpu"
  },
  "detection": {
    "confidence_threshold": 0.5,
    "target_classes": [0, 6]
  },
  "video_optimization": {
    "target_fps": 20,
    "max_width": 1280,
    "max_height": 540,
    "keep_width_native": true
  },
  "re_identification": {
    "enabled": true,
    "max_distance": 0.5,
    "max_age": 10,
    "min_hits": 8,
    "iou_threshold": 0.3
  },
  "train_number_ocr": {
    "enabled": true,
    "engine": "easyocr",
    "frame_skip": 60,
    "use_bottom_right_quadrant": true
  },
  "processing": {
    "show_preview": true,
    "save_results": true,
    "output_dir": "results"
  }
}
```

### Пример 3: С цветовыми фильтрами

```json
{
  "input_file": "video.mp4",
  "yolo": {
    "model": "yolo11m.pt",
    "device": "cpu"
  },
  "detection": {
    "confidence_threshold": 0.5,
    "target_classes": [0, 6]
  },
  "color_filters": {
    "enabled": true,
    "train": {
      "anti_color_hsv": [90, 50, 30],
      "anti_color_range": 30,
      "anti_match_threshold": 0.4
    }
  },
  "debug": {
    "show_filtered_objects": true,
    "log_detection_details": true
  },
  "processing": {
    "show_preview": true,
    "save_results": true
  }
}
```

## Поддерживаемые форматы

**Изображения:**
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Видео:**
- `.mp4`, `.avi`, `.mov`, `.mkv`

## Производительность

### Рекомендации для CPU:
- Используйте `yolo11n.pt` для максимальной скорости
- Установите `target_fps: 10-15`
- Уменьшите `max_width` и `max_height`
- Отключите `half_precision`
- Увеличьте `frame_skip` для OCR

### Рекомендации для GPU:
- Используйте `yolo11m.pt` или `yolo11l.pt`
- Установите `device: "cuda"`
- Включите `half_precision: true`
- Можно использовать большие разрешения

## Устранение неполадок

### Ошибка "Invalid CUDA device"
Убедитесь, что в `config.json` указано `"device": "cpu"` если CUDA недоступна, или `"device": "cuda"` если GPU доступен.

### Низкая производительность
- Уменьшите разрешение в `video_optimization`
- Используйте модель `yolo11n.pt`
- Увеличьте `frame_skip` или уменьшите `target_fps`
- Отключите `re_identification` и `train_number_ocr` если не нужны

### OCR не распознает номера
- Увеличьте `frame_skip` для проверки более качественных кадров
- Проверьте, что номер находится в правой половине кадра (если `use_bottom_right_quadrant: true`)
- Попробуйте другой OCR движок

## Лицензия

Проект создан для образовательных целей.

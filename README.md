# Детекция людей и поездов

Система детекции объектов (люди и поезда) на изображениях и видео с использованием YOLO v4.

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Скачайте веса YOLO v4:
   - Веса: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
   - Конфигурация: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

Поместите файлы `yolov4.weights` и `yolov4.cfg` в корневую папку проекта.

## Использование

1. Откройте `config.json` и укажите путь к вашему файлу:
```json
{
  "input_file": "C:/Users/Тихон/Desktop/my_photo.jpg"
}
```

2. Запустите:
```bash
python main.py
```

3. Откроется окно с результатом детекции. Нажмите любую клавишу (для изображения) или 'q' (для видео) для закрытия.

## Настройки

В `config.json` можно изменить:
- `confidence_threshold` - порог уверенности (0.0 - 1.0), по умолчанию 0.5
- `nms_threshold` - порог NMS (0.0 - 1.0), по умолчанию 0.4

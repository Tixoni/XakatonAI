"""
Детекция людей и поездов на изображениях и видео с использованием YOLO v4
Укажите путь к файлу в config.json и запустите этот скрипт
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path


class YOLODetector:
    """Класс для детекции объектов с использованием YOLO v4"""
    
    CLASSES = {
        0: "person",
        7: "train"
    }
    
    COLORS = {
        0: (0, 255, 0),  # Зеленый для людей
        7: (255, 0, 0),  # Синий для поездов
    }
    
    def __init__(self, weights_path="yolov4.weights", config_path="yolov4.cfg", 
                 conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        print("Загрузка YOLO сети...")
        if not os.path.exists(weights_path):
            print(f"Ошибка: файл весов {weights_path} не найден!")
            print("Скачайте: https://github.com/AlexeyAB/darknet/releases")
            sys.exit(1)
            
        if not os.path.exists(config_path):
            print(f"Ошибка: конфигурационный файл {config_path} не найден!")
            sys.exit(1)
        
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Использование CPU (самый надежный вариант)
        # Стандартный opencv-python не поддерживает CUDA
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Используется CPU")
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        print("YOLO сеть загружена успешно!")
    
    def detect(self, frame):
        """Детекция объектов на кадре"""
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id in self.CLASSES and confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append((
                    class_ids[i],
                    confidences[i],
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][2],
                    boxes[i][3]
                ))
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Отрисовка детекций на кадре"""
        result_frame = frame.copy()
        
        for class_id, confidence, x, y, w, h in detections:
            color = self.COLORS.get(class_id, (0, 0, 255))
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{self.CLASSES[class_id]}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y, label_size[1] + 10)
            cv2.rectangle(result_frame, (x, label_y - label_size[1] - 10),
                         (x + label_size[0], label_y), color, -1)
            cv2.putText(result_frame, label, (x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame


def load_config():
    """Загрузка конфигурации"""
    if not os.path.exists("config.json"):
        print("Ошибка: файл config.json не найден!")
        sys.exit(1)
    
    with open("config.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def imread_unicode(image_path):
    """Чтение изображения с поддержкой кириллицы в пути"""
    # Нормализация пути
    image_path = os.path.normpath(image_path)
    
    # Для путей с кириллицей используем чтение через байты
    # Это более надежный способ для Windows
    try:
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except Exception as e:
        print(f"Ошибка при чтении через байты: {e}")
    
    # Пробуем обычный способ как fallback
    try:
        image = cv2.imread(image_path)
        if image is not None:
            return image
    except:
        pass
    
    return None


def process_image(image_path, detector, config):
    """Обработка изображения"""
    print(f"Загрузка изображения: {image_path}")
    if not os.path.exists(image_path):
        print(f"Ошибка: файл не существует!")
        print(f"Проверьте путь: {image_path}")
        return
    
    image = imread_unicode(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение!")
        print(f"Убедитесь, что файл является корректным изображением")
        print(f"Путь: {image_path}")
        return
    
    print(f"Изображение загружено: {image.shape[1]}x{image.shape[0]} пикселей")
    
    print("Выполняется детекция...")
    detections = detector.detect(image)
    
    # Фильтруем только людей и поездов
    target_classes = config.get("detection", {}).get("target_classes", [0, 7])
    filtered = [d for d in detections if d[0] in target_classes]
    
    print(f"Найдено объектов: {len(filtered)}")
    for class_id, confidence, x, y, w, h in filtered:
        print(f"  - {detector.CLASSES[class_id]}: {confidence:.2f}")
    
    result_image = detector.draw_detections(image, filtered)
    
    # Показываем результат
    cv2.imshow('Детекция объектов', result_image)
    print("\nНажмите любую клавишу для закрытия окна...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, detector, config):
    """Обработка видео"""
    # Для путей с кириллицей пробуем разные способы
    cap = cv2.VideoCapture(video_path)
    
    # Проверяем, открылось ли видео
    if not cap.isOpened():
        # Пробуем альтернативный способ - использовать числовой индекс или другой формат пути
        try:
            # Пробуем с преобразованием пути
            video_path_alt = str(Path(video_path).absolute())
            cap = cv2.VideoCapture(video_path_alt)
        except:
            pass
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        print("Убедитесь, что путь указан правильно и файл существует")
        print("Попробуйте использовать путь без кириллицы или используйте относительный путь")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Видео: {width}x{height}, FPS: {fps}")
    print("\nНажмите 'q' для выхода\n")
    
    target_classes = config.get("detection", {}).get("target_classes", [0, 7])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        filtered = [d for d in detections if d[0] in target_classes]
        result_frame = detector.draw_detections(frame, filtered)
        
        cv2.imshow('Детекция объектов', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Главная функция"""
    print("=" * 60)
    print("Детекция людей и поездов")
    print("=" * 60)
    
    # Загрузка конфига
    config = load_config()
    
    # Определение входного файла
    input_file = config.get("input_file", "")
    if not input_file:
        print("Ошибка: в config.json не указан input_file!")
        print("Укажите путь к изображению или видео в config.json")
        sys.exit(1)
    
    # Нормализация пути (замена обратных слешей и удаление кавычек)
    input_file = input_file.strip().strip('"').strip("'")
    input_file = os.path.normpath(input_file)
    
    # Проверка существования файла
    if not os.path.exists(input_file):
        print(f"Ошибка: файл не найден!")
        print(f"Указанный путь: {input_file}")
        print(f"Абсолютный путь: {os.path.abspath(input_file)}")
        sys.exit(1)
    
    # Загрузка параметров
    yolo_config = config.get("yolo", {})
    weights_path = yolo_config.get("weights_path", "yolov4.weights")
    config_path = yolo_config.get("config_path", "yolov4.cfg")
    
    detection_config = config.get("detection", {})
    conf_threshold = detection_config.get("confidence_threshold", 0.5)
    nms_threshold = detection_config.get("nms_threshold", 0.4)
    
    # Инициализация детектора
    detector = YOLODetector(weights_path, config_path, conf_threshold, nms_threshold)
    
    # Определение типа файла
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    
    file_ext = Path(input_file).suffix.lower()
    
    if file_ext in image_extensions:
        print(f"\nОбработка изображения: {input_file}")
        process_image(input_file, detector, config)
    elif file_ext in video_extensions:
        print(f"\nОбработка видео: {input_file}")
        process_video(input_file, detector, config)
    else:
        print(f"Ошибка: неподдерживаемый формат файла {input_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()


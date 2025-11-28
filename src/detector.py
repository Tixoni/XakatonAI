"""
Модуль для детекции объектов с использованием YOLO11
"""

import sys
import cv2
from ultralytics import YOLO


class YOLODetector:
    """Класс для детекции объектов с использованием YOLO11"""
    
    CLASSES = {
        0: "person",
        6: "train"
    }
    
    DEFAULT_COLORS = {
        0: (0, 255, 0),  # Зеленый для людей (BGR)
        6: (255, 0, 0),  # Синий для поездов (BGR)
    }
    
    def __init__(self, model_path="yolo11m.pt", conf_threshold=0.5, device="cpu",
                 custom_colors=None, half_precision=False):
        """
        Инициализация детектора YOLO11
        
        Args:
            model_path: путь к модели YOLO11 (yolo11n.pt - nano, самый быстрый)
            conf_threshold: порог уверенности
            device: устройство для обработки (cpu, cuda и т.д.)
            custom_colors: словарь пользовательских цветов
            half_precision: использовать половинную точность
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.half_precision = half_precision
        
        print(f"Загрузка модели YOLO11: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            print(f"Модель загружена успешно!")
            print(f"Используется устройство: {device.upper()}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Модель будет автоматически скачана при первом запуске")
            sys.exit(1)
        
        # Подготовка цветов
        self.colors = dict(self.DEFAULT_COLORS)
        if custom_colors:
            self._apply_custom_colors(custom_colors)
    
    def _apply_custom_colors(self, colors_config):
        """Применение пользовательских цветов"""
        for key, value in colors_config.items():
            class_id = self._resolve_class_id(key)
            color_tuple = self._parse_color(value)
            
            if class_id is None:
                print(f"Предупреждение: неизвестный класс '{key}' в настройках цветов. Пропущено.")
                continue
            if color_tuple is None:
                print(f"Предупреждение: некорректный цвет для '{key}'. Ожидался формат [B,G,R]. Пропущено.")
                continue
            
            self.colors[class_id] = color_tuple
    
    @classmethod
    def _resolve_class_id(cls, key):
        """Определение ID класса по имени или числу"""
        if isinstance(key, int):
            return key if key in cls.CLASSES else None
        if isinstance(key, str):
            key = key.strip().lower()
            if key.isdigit():
                cid = int(key)
                return cid if cid in cls.CLASSES else None
            for cid, name in cls.CLASSES.items():
                if name.lower() == key:
                    return cid
        return None
    
    @staticmethod
    def _parse_color(value):
        """Преобразование цвета в кортеж BGR"""
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                b, g, r = [int(max(0, min(255, v))) for v in value]
                return (b, g, r)
            except (ValueError, TypeError):
                return None
        return None
    
    def detect(self, frame, target_classes=None):
        """
        Детекция объектов на кадре
        
        Args:
            frame: изображение (numpy array)
            target_classes: список классов для детекции (None = все)
            
        Returns:
            список детекций: [(class_id, confidence, x1, y1, x2, y2), ...]
        """
        # Запускаем детекцию
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            half=self.half_precision
        )
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Фильтруем только нужные классы
                if target_classes is None or class_id in target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((
                        class_id,
                        confidence,
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ))
        
        return detections
    
    def draw_detections(self, frame, detections, show_track_ids=False, train_numbers=None):
        """
        Отрисовка детекций на кадре
        
        Args:
            frame: кадр для отрисовки
            detections: список детекций
                - Без трекинга: [(class_id, confidence, x1, y1, x2, y2), ...]
                - С трекингом: [(track_id, class_id, confidence, x1, y1, x2, y2), ...]
            show_track_ids: показывать ли ID треков (если True, ожидается формат с track_id)
            train_numbers: словарь {track_id: train_number} для отображения номеров поездов
        """
        result_frame = frame.copy()
        
        for det in detections:
            # Определяем формат детекции
            if show_track_ids and len(det) == 7:
                # Формат с трекингом: (track_id, class_id, confidence, x1, y1, x2, y2)
                track_id, class_id, confidence, x1, y1, x2, y2 = det
            elif len(det) == 6:
                # Формат без трекинга: (class_id, confidence, x1, y1, x2, y2)
                class_id, confidence, x1, y1, x2, y2 = det
                track_id = None
            else:
                continue
            
            color = self.colors.get(class_id, (0, 0, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Формируем подпись
            class_name = self.CLASSES.get(class_id, 'unknown')
            if show_track_ids and track_id is not None:
                label = f"ID:{track_id} {class_name}: {confidence:.2f}"
                # Добавляем номер поезда, если есть
                if train_numbers and track_id in train_numbers and train_numbers[track_id]:
                    label += f" №{train_numbers[track_id]}"
            else:
                label = f"{class_name}: {confidence:.2f}"
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_height = label_size[1] + baseline
            
            # Размещаем текст сверху рамки (выше y1)
            # Если рамка слишком близко к верху экрана, размещаем внутри, но стараемся сверху
            label_y = y1 - 5  # 5 пикселей выше верхней границы рамки
            
            # Если текст выходит за верхний край экрана, размещаем его внутри рамки
            if label_y < label_height:
                label_y = y1 + label_height + 5
            
            # Фон для текста (прямоугольник за текстом)
            bg_y1 = label_y - label_height
            bg_y2 = label_y
            cv2.rectangle(result_frame, (x1, bg_y1),
                         (x1 + label_size[0], bg_y2), color, -1)
            
            # Текст
            cv2.putText(result_frame, label, (x1, label_y - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame

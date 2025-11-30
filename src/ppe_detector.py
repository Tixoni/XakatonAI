
import sys
import cv2
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict


class PPEDetector:
    
    # Классы PPE (из data_custom.yaml и main.py)
    PPE_CLASSES = {
        "hat": "helmet",
        "helmet": "helmet",
        "vest": "safety_vest",
        "safety-vest": "safety_vest",
        "Safety-Vest": "safety_vest",
        "gloves": "gloves",
        "Gloves": "gloves",
        "glass": "safety_glasses",
        "Glass": "safety_glasses"
    }
    
    
    DEFAULT_COLORS = {
        "helmet": (0, 255, 255),        # Желтый для касок (BGR)
        "safety_vest": (255, 165, 0),   # Оранжевый для жилетов (BGR)
        "gloves": (255, 0, 255),        # Пурпурный для перчаток (BGR)
        "safety_glasses": (0, 255, 0),  # Зеленый для очков (BGR)
    }
    
    def __init__(self, model_path=None, 
                 conf_threshold=0.5, device="cpu", custom_colors=None, half_precision=False):

        self.conf_threshold = conf_threshold
        self.device = device
        self.half_precision = half_precision
        self.model = None
        self.class_names = {}  # Имена классов из модели
        
        if model_path is None:
            print("Путь к модели PPE не указан, детекция PPE будет отключена")
            self.model = None
            return
        
        print(f"Загрузка модели PPE: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            # Получаем имена классов из модели
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Модель PPE загружена успешно!")
                print(f"Классы в модели: {list(self.class_names.values())}")
            else:
                print("Предупреждение: не удалось получить имена классов из модели")
                self.class_names = {}
        except Exception as e:
            print(f"Ошибка при загрузке модели PPE: {e}")
            print("Детекция PPE будет отключена")
            self.model = None
        
        # Подготовка цветов
        self.colors = dict(self.DEFAULT_COLORS)
        if custom_colors:
            self._apply_custom_colors(custom_colors)
    
    def _apply_custom_colors(self, colors_config):
        
        for key, value in colors_config.items():
            color_tuple = self._parse_color(value)
            if color_tuple is None:
                print(f"Предупреждение: некорректный цвет для '{key}'. Ожидался формат [B,G,R]. Пропущено.")
                continue
            
            # Нормализуем имя класса
            normalized_key = self._normalize_class_name(key)
            if normalized_key:
                self.colors[normalized_key] = color_tuple
    
    @staticmethod
    def _parse_color(value):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                b, g, r = [int(max(0, min(255, v))) for v in value]
                return (b, g, r)
            except (ValueError, TypeError):
                return None
        return None
    
    def _normalize_class_name(self, class_name: str) -> Optional[str]:

        class_name_lower = class_name.lower().strip()
        
        # Проверяем прямое соответствие
        if class_name_lower in self.PPE_CLASSES:
            return self.PPE_CLASSES[class_name_lower]
        
        # Проверяем через словарь
        for key, normalized in self.PPE_CLASSES.items():
            if key.lower() == class_name_lower:
                return normalized
        
        return None
    
    def detect(self, frame, target_classes=None):

        if self.model is None:
            return []
        
        # Запускаем детекцию
        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
                half=self.half_precision
            )
        except Exception as e:
            print(f"Ошибка при детекции PPE: {e}")
            return []
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Получаем имя класса из модели
                if class_id in self.class_names:
                    model_class_name = self.class_names[class_id]
                    # Нормализуем имя класса
                    normalized_name = self._normalize_class_name(model_class_name)
                    
                    if normalized_name:
                        # Фильтруем по целевым классам, если указаны
                        if target_classes is None or normalized_name in target_classes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append((
                                normalized_name,  # Используем нормализованное имя
                                confidence,
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2)
                            ))
        
        return detections
    
    def draw_detections(self, frame, detections):

        result_frame = frame.copy()
        
        for det in detections:
            if len(det) != 6:
                continue
            
            class_name, confidence, x1, y1, x2, y2 = det
            
            # Получаем цвет для класса
            color = self.colors.get(class_name, (0, 0, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Формируем подпись
            label = f"{class_name}: {confidence:.2f}"
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_height = label_size[1] + baseline
            
            # Размещаем текст сверху рамки
            label_y = y1 - 5
            if label_y < label_height:
                label_y = y1 + label_height + 5
            
            # Фон для текста
            bg_y1 = label_y - label_height
            bg_y2 = label_y
            cv2.rectangle(result_frame, (x1, bg_y1),
                         (x1 + label_size[0], bg_y2), color, -1)
            
            # Текст
            cv2.putText(result_frame, label, (x1, label_y - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame
    
    def is_enabled(self) -> bool:
        return self.model is not None


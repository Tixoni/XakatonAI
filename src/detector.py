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
        0: (0, 255, 0),
        6: (255, 0, 0),
    }
    
    def __init__(self, model_path="yolo11m.pt", conf_threshold=0.5, device="cpu",
                 custom_colors=None, half_precision=False):
        self.conf_threshold = conf_threshold
        self.device = device
        self.half_precision = half_precision
        
        print(f"Загрузка модели YOLO11: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"Модель загружена успешно на {device.upper()}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            sys.exit(1)
            
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
        # ... (Код метода detect) ...
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
    
    def draw_detections(self, frame, detections):
        # ... (Код метода draw_detections) ...
        result_frame = frame.copy()
        for class_id, confidence, x1, y1, x2, y2 in detections:
            color = self.colors.get(class_id, (0, 0, 255))
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            # ... отрисовка текста ...
        return result_frame
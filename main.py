"""
Детекция людей и поездов на изображениях и видео с использованием YOLOv11
Оптимизировано для работы на CPU
"""

import cv2
import json
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np


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
        Инициализация детектора YOLOv8
        
        Args:
            model_path: путь к модели YOLOv8 (yolov8n.pt - nano, самый быстрый)
            conf_threshold: порог уверенности
            device: устройство для обработки (cpu, cuda и т.д.)
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
    
    def draw_detections(self, frame, detections):
        """Отрисовка детекций на кадре"""
        result_frame = frame.copy()
        
        for class_id, confidence, x1, y1, x2, y2 in detections:
            color = self.colors.get(class_id, (0, 0, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Подпись
            label = f"{self.CLASSES.get(class_id, 'unknown')}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1, label_size[1] + 10)
            
            # Фон для текста
            cv2.rectangle(result_frame, (x1, label_y - label_size[1] - 10),
                         (x1 + label_size[0], label_y), color, -1)
            
            # Текст
            cv2.putText(result_frame, label, (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame


def load_config():
    """Загрузка конфигурации"""
    if not os.path.exists("config.json"):
        print("Ошибка: файл config.json не найден!")
        sys.exit(1)
    
    with open("config.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def resize_frame(frame, max_width, max_height, maintain_aspect=True, keep_width_native=False):
    """
    Изменение размера кадра с сохранением пропорций
    
    Args:
        frame: кадр изображения
        max_width: максимальная ширина
        max_height: максимальная высота
        maintain_aspect: сохранять пропорции
        
    Returns:
        измененный кадр, масштаб
    """
    height, width = frame.shape[:2]
    
    # Ограничиваем только высоту, оставляя ширину нативной
    if keep_width_native:
        if max_height and height > max_height:
            new_height = max_height
            resized = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)
            scale = new_height / height
            return resized, scale
        return frame, 1.0
    
    if (max_width == 0 or max_width is None) and (max_height == 0 or max_height is None):
        return frame, 1.0
    
    if max_width == 0 or max_width is None:
        max_width = width
    if max_height == 0 or max_height is None:
        max_height = height
    
    if width <= max_width and height <= max_height:
        return frame, 1.0
    
    if maintain_aspect:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = max_width
        new_height = max_height
        scale = min(max_width / width, max_height / height)
    
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def imread_unicode(image_path):
    """Чтение изображения с поддержкой кириллицы в пути"""
    image_path = os.path.normpath(image_path)
    
    # Для путей с кириллицей используем чтение через байты
    try:
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except Exception as e:
        print(f"Ошибка при чтении через байты: {e}")
    
    # Пробуем обычный способ
    try:
        image = cv2.imread(image_path)
        if image is not None:
            return image
    except:
        pass
    
    return None


def resolve_filter_class_id(class_key, class_map):
    """Получение ID класса по имени или числу"""
    if isinstance(class_key, int):
        return class_key if class_key in class_map else None
    if isinstance(class_key, str):
        key = class_key.strip().lower()
        if key.isdigit():
            cid = int(key)
            return cid if cid in class_map else None
        for cid, name in class_map.items():
            if name.lower() == key:
                return cid
    return None


def parse_color_range(range_values):
    """Подготовка нижних и верхних границ цвета"""
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 3:
        return None
    try:
        return np.array([int(max(0, min(255, v))) for v in range_values], dtype=np.uint8)
    except (ValueError, TypeError):
        return None


def apply_color_filters(frame, detections, filters_cfg, class_map, debug_cfg=None):
    """
    Применяет цветовые фильтры и возвращает отфильтрованные и отклонённые детекции
    """
    if not filters_cfg or not filters_cfg.get("enabled", False):
        return detections, []
    
    filtered = []
    rejected = []
    log_details = bool(debug_cfg.get("log_detection_details")) if debug_cfg else False
    
    for det in detections:
        class_id, conf, x1, y1, x2, y2 = det
        class_name = class_map.get(class_id, str(class_id))
        filter_cfg = None
        
        # Поиск конфигурации фильтра по имени или ID
        for key, cfg in filters_cfg.items():
            if key == "enabled":
                continue
            cid = resolve_filter_class_id(key, class_map)
            if cid is not None and cid == class_id:
                filter_cfg = cfg
                break
        
        if not filter_cfg:
            filtered.append(det)
            continue
        
        roi = extract_roi(frame, x1, y1, x2, y2)
        if roi is None:
            rejected.append((det, "empty_roi"))
            continue
        
        passed, info = passes_color_filter(roi, filter_cfg)
        
        if log_details:
            reason = info.get("reason", "pass")
            match_ratio = info.get("match_ratio")
            anti_ratio = info.get("anti_ratio")
            details = []
            if match_ratio is not None:
                details.append(f"match={match_ratio:.2f}/{info.get('match_threshold', '-')}")
            if anti_ratio is not None:
                details.append(f"anti={anti_ratio:.2f}/{info.get('anti_threshold', '-')}")
            detail_text = " | ".join(details)
            status = "PASS" if passed else f"FILTERED({reason})"
            print(f"[DEBUG] {class_name} ({conf:.2f}) -> {status} {detail_text}")
        
        if passed:
            filtered.append(det)
        else:
            rejected.append({
                "det": det,
                "info": info
            })
    
    return filtered, rejected


def annotate_rejected(frame, rejected, detector, color=(0, 0, 255)):
    """Возвращает кадр с выделенными отклонёнными объектами"""
    overlay = frame.copy()
    for item in rejected:
        det = item.get("det")
        info = item.get("info", {})
        if not det:
            continue
        class_id, _, x1, y1, x2, y2 = det
        class_name = detector.CLASSES.get(class_id, f"class_{class_id}")
        reason = info.get("reason", "filtered")
        label = f"{class_name} [{reason}]"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return overlay


def extract_roi(frame, x1, y1, x2, y2):
    """Вырезает область интереса с учетом границ кадра"""
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def passes_color_filter(roi, cfg):
    """Проверяет, удовлетворяет ли ROI заданным цветовым фильтрам"""
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return False, {"reason": "empty_roi"}
    
    info = {}
    positive_mask = None
    
    # RGB диапазон
    if "min_rgb" in cfg and "max_rgb" in cfg:
        lower_rgb = parse_color_range(cfg["min_rgb"])
        upper_rgb = parse_color_range(cfg["max_rgb"])
        if lower_rgb is not None and upper_rgb is not None:
            rgb_mask = cv2.inRange(roi, lower_rgb, upper_rgb)
            positive_mask = rgb_mask if positive_mask is None else cv2.bitwise_and(positive_mask, rgb_mask)
    
    # HSV диапазон
    if "min_hsv" in cfg and "max_hsv" in cfg:
        lower_hsv = parse_color_range(cfg["min_hsv"])
        upper_hsv = parse_color_range(cfg["max_hsv"])
        if lower_hsv is not None and upper_hsv is not None:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
            positive_mask = hsv_mask if positive_mask is None else cv2.bitwise_and(positive_mask, hsv_mask)
    
    positive_threshold = float(cfg.get("match_threshold", 0.1))
    positive_threshold = max(0.0, min(1.0, positive_threshold))
    
    if positive_mask is not None:
        match_ratio = cv2.countNonZero(positive_mask) / total_pixels
        info["match_ratio"] = match_ratio
        info["match_threshold"] = positive_threshold
        if match_ratio < positive_threshold:
            info["reason"] = "positive_miss"
            return False, info
    
    # Anti-color фильтр в HSV
    anti_threshold = float(cfg.get("anti_match_threshold", cfg.get("match_threshold", 0.1)))
    anti_threshold = max(0.0, min(1.0, anti_threshold))
    info["anti_threshold"] = anti_threshold
    
    anti_ratio = None
    if "anti_color_hsv" in cfg:
        center = parse_color_range(cfg["anti_color_hsv"])
        if center is not None:
            range_val = int(cfg.get("anti_color_range", 20))
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([
                max(0, center[0] - range_val),
                max(0, center[1] - range_val),
                max(0, center[2] - range_val)
            ], dtype=np.uint8)
            upper = np.array([
                min(180, center[0] + range_val),
                min(255, center[1] + range_val),
                min(255, center[2] + range_val)
            ], dtype=np.uint8)
            anti_mask = cv2.inRange(hsv_roi, lower, upper)
            anti_ratio = cv2.countNonZero(anti_mask) / total_pixels
    
    if "anti_color_rgb" in cfg and anti_ratio is None:
        center = parse_color_range(cfg["anti_color_rgb"])
        if center is not None:
            range_val = int(cfg.get("anti_color_range", 20))
            lower = np.array([
                max(0, center[0] - range_val),
                max(0, center[1] - range_val),
                max(0, center[2] - range_val)
            ], dtype=np.uint8)
            upper = np.array([
                min(255, center[0] + range_val),
                min(255, center[1] + range_val),
                min(255, center[2] + range_val)
            ], dtype=np.uint8)
            anti_mask = cv2.inRange(roi, lower, upper)
            anti_ratio = cv2.countNonZero(anti_mask) / total_pixels
    
    if anti_ratio is not None:
        info["anti_ratio"] = anti_ratio
        if anti_ratio >= anti_threshold:
            info["reason"] = "anti_match"
            return False, info
    
    info["reason"] = "pass"
    return True, info


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
        return
    
    print(f"Изображение загружено: {image.shape[1]}x{image.shape[0]} пикселей")
    
    processing_cfg = config.get("processing", {})
    show_preview = processing_cfg.get("show_preview", True)
    save_results = processing_cfg.get("save_results", False)
    output_dir = processing_cfg.get("output_dir", "results")
    debug_cfg = config.get("debug", {})
    
    # Оптимизация размера если нужно
    opt_config = config.get("video_optimization", {})
    max_width = opt_config.get("max_width", 1920)
    max_height = opt_config.get("max_height", 1080)
    maintain_aspect = opt_config.get("maintain_aspect_ratio", True)
    keep_width_native = opt_config.get("keep_width_native", False)
    
    needs_resize = False
    if keep_width_native and max_height and image.shape[0] > max_height:
        needs_resize = True
    elif (image.shape[1] > max_width or image.shape[0] > max_height):
        needs_resize = True
    
    if needs_resize:
        target_desc = f"до высоты {max_height}px" if keep_width_native else f"до {max_width}x{max_height}"
        print(f"Оптимизация размера {target_desc} ...")
        image, scale = resize_frame(image, max_width, max_height, maintain_aspect, keep_width_native)
        print(f"Размер изменен: {image.shape[1]}x{image.shape[0]} пикселей (масштаб: {scale:.2f})")
    
    print("Выполняется детекция...")
    
    target_classes = config.get("detection", {}).get("target_classes", [0, 6])
    detections = detector.detect(image, target_classes=target_classes)
    
    color_filters = config.get("color_filters", {})
    detections, rejected = apply_color_filters(image, detections, color_filters, detector.CLASSES, debug_cfg)
    
    print(f"Найдено объектов: {len(detections)}")
    if rejected:
        print(f"Отклонено цветовым фильтром: {len(rejected)}")
    for class_id, confidence, x1, y1, x2, y2 in detections:
        class_name = detector.CLASSES.get(class_id, "unknown")
        print(f"  - {class_name}: {confidence:.2f}")
    
    result_image = detector.draw_detections(image, detections)
    
    if debug_cfg.get("show_filtered_objects") and rejected:
        result_image = annotate_rejected(result_image, rejected, detector)
    
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{Path(image_path).stem}_detected.png"
        cv2.imwrite(str(output_path), result_image)
        print(f"Результат сохранен: {output_path}")
    
    if show_preview:
        cv2.imshow('Детекция объектов', result_image)
        print("\nНажмите любую клавишу для закрытия окна...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path, detector, config):
    """Обработка видео с оптимизацией"""
    # Для путей с кириллицей
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        try:
            video_path_alt = str(Path(video_path).absolute())
            cap = cv2.VideoCapture(video_path_alt)
        except:
            pass
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        print("Убедитесь, что путь указан правильно и файл существует")
        return
    
    # Получаем параметры видео
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Оригинальное видео: {width}x{height}, FPS: {original_fps}, Кадров: {total_frames}")
    
    # Параметры оптимизации
    opt_config = config.get("video_optimization", {})
    target_fps = opt_config.get("target_fps", 10)
    max_width = opt_config.get("max_width", 1280)
    max_height = opt_config.get("max_height", 720)
    frame_skip = opt_config.get("frame_skip", 1)
    maintain_aspect = opt_config.get("maintain_aspect_ratio", True)
    keep_width_native = opt_config.get("keep_width_native", False)
    
    processing_cfg = config.get("processing", {})
    show_preview = processing_cfg.get("show_preview", True)
    save_results = processing_cfg.get("save_results", False)
    output_dir = processing_cfg.get("output_dir", "results")
    debug_cfg = config.get("debug", {})
    
    # Вычисляем, через сколько кадров обрабатывать
    # Если frame_skip явно указан > 1, используем его, иначе вычисляем на основе target_fps
    if frame_skip > 1:
        skip_frames = frame_skip
        print(f"Используется явный пропуск кадров: {skip_frames}")
    else:
        # Вычисляем пропуск на основе целевого FPS
        if target_fps <= 0 or original_fps <= 0:
            skip_frames = 1
        elif target_fps < original_fps:
            skip_frames = max(1, int(original_fps / target_fps))
        else:
            skip_frames = 1
        print(f"Автоматический пропуск кадров для FPS {target_fps}: {skip_frames}")
    
    # Предварительная оценка размера вывода
    temp_frame = cap.read()[1]
    if temp_frame is None:
        print("Не удалось прочитать первый кадр.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    resized_sample, _ = resize_frame(temp_frame, max_width, max_height, maintain_aspect, keep_width_native)
    display_width = resized_sample.shape[1]
    display_height = resized_sample.shape[0]
    
    print(f"Обработка: пропуск кадров {skip_frames}, размер {display_width}x{display_height}")
    print(f"Целевой FPS: {target_fps}")
    print("\nНажмите 'q' для выхода\n")
    
    target_classes = config.get("detection", {}).get("target_classes", [0, 6])
    detections_count = {cid: 0 for cid in target_classes}
    color_filters = config.get("color_filters", {})
    rejected_total = 0
    writer = None
    output_path = None
    
    frame_count = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Пропускаем кадры для снижения FPS
            if frame_count % skip_frames != 0:
                continue
            
            processed_count += 1
            
            resize_needed = False
            if keep_width_native:
                if max_height and frame.shape[0] > max_height:
                    resize_needed = True
            else:
                if frame.shape[1] > max_width or frame.shape[0] > max_height:
                    resize_needed = True
            
            if resize_needed:
                resized_frame, _ = resize_frame(frame, max_width, max_height, maintain_aspect, keep_width_native)
            else:
                resized_frame = frame
            
            # Детекция
            detections = detector.detect(resized_frame, target_classes=target_classes)
            detections, rejected = apply_color_filters(resized_frame, detections, color_filters, detector.CLASSES, debug_cfg)
            rejected_total += len(rejected)
            
            # Подсчет
            for class_id, _, _, _, _, _ in detections:
                if class_id in detections_count:
                    detections_count[class_id] += 1
            
            # Отрисовка
            result_frame = detector.draw_detections(resized_frame, detections)
            if debug_cfg.get("show_filtered_objects") and rejected:
                result_frame = annotate_rejected(result_frame, rejected, detector, color=(0, 0, 255))
            
            # Информация на кадре
            counts_text = " | ".join(
                f"{detector.CLASSES.get(cid, f'class_{cid}')}: {detections_count[cid]}"
                for cid in detections_count
            )
            info_text = f"Кадр: {processed_count}" + (f" | {counts_text}" if counts_text else "")
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Сохранение результата
            if save_results:
                if writer is None:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = Path(output_dir) / f"{Path(video_path).stem}_detected.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps,
                                             (result_frame.shape[1], result_frame.shape[0]))
                writer.write(result_frame)
            
            # Показываем результат
            if show_preview:
                cv2.imshow('Детекция объектов', result_frame)
                
                # Задержка для соответствия целевому FPS
                delay = int(1000 / target_fps) if target_fps > 0 else 1
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
            
            # Прогресс
            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Обработано кадров: {processed_count}, прогресс: {progress:.1f}%")
    
    except KeyboardInterrupt:
        print("\nОбработка прервана")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nОбработка завершена!")
        print(f"Обработано кадров: {processed_count}")
        for cid, count in detections_count.items():
            class_name = detector.CLASSES.get(cid, f"class_{cid}")
            print(f"Обнаружено {class_name}: {count}")
        if rejected_total:
            print(f"Отклонено цветовым фильтром: {rejected_total}")
        if save_results and output_path:
            print(f"Видео сохранено: {output_path}")


def main():
    """Главная функция"""
    print("=" * 60)
    print("Детекция людей и поездов (YOLO11)")
    print("=" * 60)
    
    # Загрузка конфига
    config = load_config()
    
    # Определение входного файла
    input_file = config.get("input_file", "")
    if not input_file:
        print("Ошибка: в config.json не указан input_file!")
        print("Укажите путь к изображению или видео в config.json")
        sys.exit(1)
    
    # Нормализация пути
    input_file = input_file.strip().strip('"').strip("'")
    input_file = os.path.normpath(input_file)
    
    if not os.path.exists(input_file):
        print(f"Ошибка: файл не найден!")
        print(f"Указанный путь: {input_file}")
        print(f"Абсолютный путь: {os.path.abspath(input_file)}")
        sys.exit(1)
    
    # Загрузка параметров
    yolo_config = config.get("yolo", {})
    model_path = yolo_config.get("model", "yolo11m.pt")
    device = yolo_config.get("device", "cpu")
    
    detection_config = config.get("detection", {})
    conf_threshold = detection_config.get("confidence_threshold", 0.5)
    custom_colors = config.get("colors", {})
    processing_cfg = config.get("processing", {})
    half_precision = processing_cfg.get("half_precision", False)
    
    # Инициализация детектора
    detector = YOLODetector(model_path, conf_threshold, device, custom_colors, half_precision)
    
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

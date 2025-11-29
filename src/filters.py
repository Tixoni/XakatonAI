"""
Модуль для цветовых фильтров детекций
"""

import cv2
import numpy as np
from src.image_utils import extract_roi


def parse_color_range(range_values):
    """Подготовка нижних и верхних границ цвета"""
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 3:
        return None
    try:
        return np.array([int(max(0, min(255, v))) for v in range_values], dtype=np.uint8)
    except (ValueError, TypeError):
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
            rejected.append({
                "det": det,
                "info": {"reason": "empty_roi"}
            })
            continue
        
        passed, info = passes_color_filter(roi, filter_cfg)
        
        # Дополнительная проверка: для поездов требуется наличие красного цвета
        if passed and class_name == "train" and filter_cfg.get("require_red_color", False):
            # Используем настройки lighting_compensation из config, если они есть
            lighting_compensation = None
            if debug_cfg:
                # Получаем настройки из config через debug_cfg, если они переданы
                # Или используем стандартные настройки для лучшего определения красного
                lighting_compensation = {"enabled": True, "normalize_brightness": False, "wider_color_ranges": True}
            
            color_info = detect_dominant_color(roi, top_n=3, lighting_compensation=lighting_compensation)
            all_percentages = color_info.get("all_percentages", {})
            top_colors = color_info.get("top_colors", [])
            
            # Проверяем наличие красного цвета (red и red2 объединяются в red)
            red_percentage = all_percentages.get("red", 0.0)
            # Также проверяем в top_colors
            for color_item in top_colors:
                if color_item.get("name") == "red":
                    red_percentage = max(red_percentage, color_item.get("percentage", 0.0))
            
            min_red_threshold = filter_cfg.get("min_red_threshold", 0.1)
            if red_percentage < min_red_threshold:
                passed = False
                info["reason"] = "no_red_color"
                info["red_percentage"] = red_percentage
                info["min_red_threshold"] = min_red_threshold
        
        # Фильтр для поездов: требуется красный цвет >= 20% и синий цвет <= 5%
        if passed and class_name == "train":
            # Используем настройки lighting_compensation для лучшего определения цветов
            lighting_compensation = None
            if debug_cfg:
                lighting_compensation = {"enabled": True, "normalize_brightness": False, "wider_color_ranges": True}
            
            color_info = detect_dominant_color(roi, top_n=5, lighting_compensation=lighting_compensation)
            all_percentages = color_info.get("all_percentages", {})
            top_colors = color_info.get("top_colors", [])
            
            # Проверяем процент красного цвета (должен быть >= 20%)
            red_percentage = all_percentages.get("red", 0.0)
            # Также проверяем в top_colors
            for color_item in top_colors:
                if color_item.get("name") == "red":
                    red_percentage = max(red_percentage, color_item.get("percentage", 0.0))
            
            min_red_threshold = 0.2  # 20%
            if red_percentage < min_red_threshold:
                passed = False
                info["reason"] = "insufficient_red_color"
                info["red_percentage"] = red_percentage
                info["min_red_threshold"] = min_red_threshold
            
            # Проверяем процент синего и голубого цвета (должен быть <= 5%)
            if passed:
                blue_percentage = all_percentages.get("blue", 0.0)
                cyan_percentage = all_percentages.get("cyan", 0.0)
                
                # Также проверяем в top_colors
                for color_item in top_colors:
                    color_name = color_item.get("name", "")
                    if color_name == "blue":
                        blue_percentage = max(blue_percentage, color_item.get("percentage", 0.0))
                    elif color_name == "cyan":
                        cyan_percentage = max(cyan_percentage, color_item.get("percentage", 0.0))
                
                # Суммируем синий и голубой цвета
                total_blue_cyan = blue_percentage + cyan_percentage
                max_blue_threshold = 0.05  # 5%
                
                if total_blue_cyan > max_blue_threshold:
                    passed = False
                    info["reason"] = "too_much_blue"
                    info["blue_percentage"] = total_blue_cyan
                    info["max_blue_threshold"] = max_blue_threshold
        
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


def detect_dominant_color(roi, top_n=2, lighting_compensation=None):
    """
    Определяет несколько преобладающих цветов объекта
    
    Args:
        roi: Область интереса (numpy array BGR)
        top_n: Количество цветов для возврата (по умолчанию 2)
    
    Returns:
        dict: {
            "top_colors": [  # Список топ цветов, отсортированных по проценту
                {"name": "red", "percentage": 0.45},
                {"name": "blue", "percentage": 0.30}
            ],
            "color_name": "red",  # Основной цвет (для обратной совместимости)
            "color_percentage": 0.45,  # Процент основного цвета
            "bgr_avg": [100, 50, 200]  # Средний BGR цвет
        }
    """
    if roi is None or roi.size == 0:
        return {"color_name": "unknown", "color_percentage": 0.0}
    
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return {"color_name": "unknown", "color_percentage": 0.0}
    
    # Преобразуем в HSV для более точного определения цвета
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Компенсация освещения: нормализация яркости
    if lighting_compensation and lighting_compensation.get("normalize_brightness", False):
        # Вычисляем среднюю яркость
        v_channel = hsv_roi[:, :, 2].astype(np.float32)
        avg_v = np.mean(v_channel)
        
        # Нормализуем яркость к среднему значению (128)
        if avg_v > 0:
            scale_factor = 128.0 / avg_v
            v_channel_normalized = np.clip(v_channel * scale_factor, 0, 255).astype(np.uint8)
            hsv_roi = hsv_roi.copy()
            hsv_roi[:, :, 2] = v_channel_normalized
    
    # Определяем основные цвета в HSV
    # Используем более широкие диапазоны для устойчивости к освещению
    wider_ranges = lighting_compensation and lighting_compensation.get("wider_color_ranges", False)
    
    if wider_ranges:
        # Более широкие диапазоны, но с минимальной насыщенностью выше порога серых
        # чтобы гарантировать, что серые не попадут в цветные диапазоны
        sat_min = 45  # Выше порога серых (40), но ниже стандартного (50)
        val_min = 30  # Вместо 50
        color_ranges = [
            ("red", ([0, sat_min, val_min], [10, 255, 255])),  # Красный (0-10)
            ("red2", ([170, sat_min, val_min], [180, 255, 255])),  # Красный (170-180)
            ("orange", ([11, sat_min, val_min], [25, 255, 255])),  # Оранжевый
            ("yellow", ([26, sat_min, val_min], [35, 255, 255])),  # Желтый
            ("green", ([36, sat_min, val_min], [85, 255, 255])),  # Зеленый
            ("cyan", ([86, sat_min, val_min], [100, 255, 255])),  # Голубой
            ("blue", ([101, sat_min, val_min], [130, 255, 255])),  # Синий
            ("purple", ([131, sat_min, val_min], [169, 255, 255])),  # Фиолетовый
        ]
    else:
        # Стандартные диапазоны (насыщенность 50, что выше порога серых 40)
        color_ranges = [
            ("red", ([0, 50, 50], [10, 255, 255])),  # Красный (0-10)
            ("red2", ([170, 50, 50], [180, 255, 255])),  # Красный (170-180)
            ("orange", ([11, 50, 50], [25, 255, 255])),  # Оранжевый
            ("yellow", ([26, 50, 50], [35, 255, 255])),  # Желтый
            ("green", ([36, 50, 50], [85, 255, 255])),  # Зеленый
            ("cyan", ([86, 50, 50], [100, 255, 255])),  # Голубой
            ("blue", ([101, 50, 50], [130, 255, 255])),  # Синий
            ("purple", ([131, 50, 50], [169, 255, 255])),  # Фиолетовый
        ]
    
    # Также проверяем яркость для белого/черного/серого
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    color_percentages = {}
    
    # СНАЧАЛА определяем оттенки серого, чтобы исключить их из цветных диапазонов
    # Используем более строгий порог насыщенности для серых (40 вместо 30)
    # чтобы гарантировать, что серые пиксели не попадут в цветные диапазоны
    gray_saturation_threshold = 40
# Маска для всех серых оттенков (низкая насыщенность)
    gray_mask_all = hsv_roi[:, :, 1] < gray_saturation_threshold
    
    # Определяем оттенки серого как отдельные категории
    # Светло-серый (высокая яркость, низкая насыщенность)
    light_gray_mask = gray_mask_all & (hsv_roi[:, :, 2] > 180) & (hsv_roi[:, :, 2] <= 240)
    light_gray_pixels = np.sum(light_gray_mask)
    if light_gray_pixels / total_pixels > 0.05:
        color_percentages["light_gray"] = light_gray_pixels / total_pixels
    
    # Темно-серый (низкая яркость, низкая насыщенность)
    dark_gray_mask = gray_mask_all & (hsv_roi[:, :, 2] >= 50) & (hsv_roi[:, :, 2] < 120)
    dark_gray_pixels = np.sum(dark_gray_mask)
    if dark_gray_pixels / total_pixels > 0.05:
        color_percentages["dark_gray"] = dark_gray_pixels / total_pixels
    
    # Средний серый
    gray_mask = gray_mask_all & (hsv_roi[:, :, 2] >= 120) & (hsv_roi[:, :, 2] <= 180)
    gray_pixels = np.sum(gray_mask)
    if gray_pixels / total_pixels > 0.05:
        color_percentages["gray"] = gray_pixels / total_pixels
    
    # Белый (очень высокая яркость, очень низкая насыщенность)
    white_mask = (hsv_roi[:, :, 1] < 20) & (hsv_roi[:, :, 2] > 240)
    white_pixels = np.sum(white_mask)
    if white_pixels / total_pixels > 0.05:
        color_percentages["white"] = white_pixels / total_pixels
    
    # Черный (очень низкая яркость, очень низкая насыщенность)
    black_mask = (hsv_roi[:, :, 1] < 20) & (hsv_roi[:, :, 2] < 50)
    black_pixels = np.sum(black_mask)
    if black_pixels / total_pixels > 0.05:
        color_percentages["black"] = black_pixels / total_pixels
    
    # Объединяем все серые маски для исключения из цветных диапазонов
    # Включаем ВСЕ пиксели с низкой насыщенностью, даже если они не попали в конкретные категории
    all_gray_mask = gray_mask_all | white_mask | black_mask
    
    # Проверяем каждый цвет, ИСКЛЮЧАЯ серые пиксели
    for color_name, (lower, upper) in color_ranges:
        lower_hsv = np.array(lower, dtype=np.uint8)
        upper_hsv = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        # Исключаем серые пиксели из цветной маски
        color_mask = color_mask & (~all_gray_mask.astype(np.uint8) * 255)
        percentage = cv2.countNonZero(color_mask) / total_pixels
        color_percentages[color_name] = percentage
    
    # Объединяем красные диапазоны
    if "red" in color_percentages and "red2" in color_percentages:
        color_percentages["red"] = color_percentages["red"] + color_percentages["red2"]
        del color_percentages["red2"]
    
    # Сортируем цвета по проценту (от большего к меньшему)
    if not color_percentages:
        top_colors = []
        dominant_color = "unknown"
        max_percentage = 0.0
    else:
        # Фильтруем цвета с минимальным процентом (больше 5%)
        filtered_colors = {k: v for k, v in color_percentages.items() if v > 0.05}
        
        # Сортируем по проценту
        sorted_colors = sorted(filtered_colors.items(), key=lambda x: x[1], reverse=True)
        
        # Берем топ-N цветов
        top_colors = [
            {"name": name, "percentage": pct}
            for name, pct in sorted_colors[:top_n]
        ]
        
        # Для обратной совместимости оставляем основной цвет
        if top_colors:
            dominant_color = top_colors[0]["name"]
            max_percentage = top_colors[0]["percentage"]
        else:
            dominant_color = "unknown"
            max_percentage = 0.0
    
    # Вычисляем средний BGR цвет
    avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
    
    return {
        "top_colors": top_colors,
        "color_name": dominant_color,  # Для обратной совместимости
        "color_percentage": max_percentage,  # Для обратной совместимости
        "bgr_avg": avg_bgr.tolist(),
        "all_percentages": color_percentages  # Все проценты для отладки
    }
def get_color_for_detection(frame, detection, top_n=2, crop_border_ratio=0.15, excluded_colors=None, lighting_compensation=None):
    """
    Определяет преобладающие цвета для конкретной детекции
    
    Args:
        frame: Кадр видео
        detection: (class_id, confidence, x1, y1, x2, y2)
        top_n: Количество цветов для возврата (по умолчанию 2)
        crop_border_ratio: Коэффициент обрезки краев ROI (0.0-0.5)
        excluded_colors: Список запрещенных цветов для этого класса (например, ["green"])
        lighting_compensation: Настройки компенсации освещения
    
    Returns:
        dict: Информация о преобладающих цветах (без запрещенных цветов)
    """
    class_id, conf, x1, y1, x2, y2 = detection
    roi = extract_roi(frame, x1, y1, x2, y2, crop_border_ratio=crop_border_ratio)
    color_info = detect_dominant_color(roi, top_n=top_n, lighting_compensation=lighting_compensation)
    
    # Фильтруем запрещенные цвета
    if excluded_colors and isinstance(excluded_colors, list):
        excluded_colors_lower = [c.lower() for c in excluded_colors]
        
        # Фильтруем из top_colors
        if "top_colors" in color_info:
            filtered_top_colors = [
                color_item for color_item in color_info["top_colors"]
                if color_item.get("name", "").lower() not in excluded_colors_lower
            ]
            color_info["top_colors"] = filtered_top_colors
            
            # Обновляем основной цвет (для обратной совместимости)
            if filtered_top_colors:
                color_info["color_name"] = filtered_top_colors[0]["name"]
                color_info["color_percentage"] = filtered_top_colors[0]["percentage"]
            else:
                # Если все цвета были исключены, возвращаем unknown
                color_info["color_name"] = "unknown"
                color_info["color_percentage"] = 0.0
        
        # Также удаляем из all_percentages для отладки
        if "all_percentages" in color_info:
            for excluded_color in excluded_colors_lower:
                if excluded_color in color_info["all_percentages"]:
                    del color_info["all_percentages"][excluded_color]
    
    return color_info


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
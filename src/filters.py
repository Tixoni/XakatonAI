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

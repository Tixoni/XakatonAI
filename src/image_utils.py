import cv2
import numpy as np
import os

def imread_unicode(image_path):
    """Чтение изображения с поддержкой кириллицы в пути"""
    image_path = os.path.normpath(image_path)
    try:
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        pass
    try:
        return cv2.imread(image_path)
    except:
        return None

def resize_frame(frame, max_width, max_height, maintain_aspect=True, keep_width_native=False):
    """Изменение размера кадра с сохранением пропорций"""
    height, width = frame.shape[:2]
    
    if keep_width_native:
        if max_height and height > max_height:
            new_height = max_height
            resized = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)
            scale = new_height / height
            return resized, scale
        return frame, 1.0
    
    if (max_width == 0 or max_width is None) and (max_height == 0 or max_height is None):
        return frame, 1.0
    
    if max_width == 0: max_width = width
    if max_height == 0: max_height = height
    
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

def extract_roi(frame, x1, y1, x2, y2, crop_border_ratio=0.0):
    """
    Вырезает область интереса
    
    Args:
        frame: кадр изображения
        x1, y1, x2, y2: координаты области
        crop_border_ratio: коэффициент обрезки краев (0.0-0.5), чтобы исключить края детекции
    """
    h, w = frame.shape[:2]
    
    # Применяем обрезку краев, если указано
    if crop_border_ratio > 0.0 and crop_border_ratio < 0.5:
        width = x2 - x1
        height = y2 - y1
        crop_x = int(width * crop_border_ratio)
        crop_y = int(height * crop_border_ratio)
        x1 = x1 + crop_x
        y1 = y1 + crop_y
        x2 = x2 - crop_x
        y2 = y2 - crop_y
    
    # Проверяем границы
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
"""
Модуль для представления объектов на экране
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import cv2
from src.image_utils import extract_roi
from src.filters import detect_dominant_color


class ObjectStatus(Enum):
    """Статусы объекта"""
    STAY = "stay"      # Остается на месте
    GO = "go"         # Движется
    WORK = "work"     # Работает (для поездов - в движении)
    UNKNOWN = "unknown"  # Неизвестно


class ScreenObject:
    """Класс для представления объекта на экране"""
    
    def __init__(self, object_id: int, object_type: str, class_id: int,
                 bbox: Tuple[int, int, int, int], confidence: float,
                 frame_num: int, features: Optional[np.ndarray] = None):
       
        self.object_id = object_id  # Уникальный ID в группе по типу
        self.object_type = object_type  # 'person', 'train', и т.д.
        self.class_id = class_id
        self.bbox = bbox
        self.confidence = confidence
        self.frame_num = frame_num
        self.features = features
        
        # История позиций для определения статуса
        self.position_history: List[Tuple[int, int, int, int]] = [bbox]
        self.frame_count = 1  # Количество кадров, которое объект фиксировался камерой
        
        # Основные цвета объекта
        self.dominant_colors: List[Tuple[int, int, int]] = []  # BGR формат (для обратной совместимости)
        self.color_info: Optional[Dict] = None  # Полная информация о цветах (названия, проценты, BGR)
        
        # Статус объекта
        self.status = ObjectStatus.UNKNOWN
        
        # Дополнительная информация
        self.train_number: Optional[str] = None  # Номер поезда (для train)
        self.last_update_frame = frame_num
        
        # Пороги для определения статуса
        self.movement_threshold = 5.0  # Минимальное перемещение для статуса GO (в пикселях)
        self.stay_threshold = 2.0  # Максимальное перемещение для статуса STAY
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float,
               frame_num: int, features: Optional[np.ndarray] = None):
        """
        Обновление объекта новыми данными
        
        Args:
            bbox: новые координаты
            confidence: новая уверенность
            frame_num: номер кадра
            features: новые признаки
        """
        self.bbox = bbox
        self.confidence = confidence
        self.frame_count += 1
        self.last_update_frame = frame_num
        
        # Обновляем историю позиций
        self.position_history.append(bbox)
        # Ограничиваем историю (храним последние 30 позиций)
        if len(self.position_history) > 30:
            self.position_history.pop(0)
        
        # Обновляем признаки
        if features is not None:
            if self.features is not None:
                alpha = 0.1  # Коэффициент обновления
                self.features = (1 - alpha) * self.features + alpha * features
            else:
                self.features = features
    
    def update_colors(self, frame: np.ndarray, top_n: int = 4, lighting_compensation: Optional[Dict] = None):
        """
        Обновляет основные цвета объекта на основе текущего кадра
        Использует улучшенный метод detect_dominant_color из filters.py
        
        Args:
            frame: кадр изображения (BGR)
            top_n: количество основных цветов для определения (по умолчанию 4)
            lighting_compensation: настройки компенсации освещения (опционально)
        """
        roi = extract_roi(frame, *self.bbox, crop_border_ratio=0.1)  # Обрезаем края на 10%
        if roi is None or roi.size == 0:
            return
        
        # Используем улучшенный метод определения цветов
        try:
            color_info = detect_dominant_color(
                roi, 
                top_n=top_n, 
                lighting_compensation=lighting_compensation
            )
            
            # Сохраняем полную информацию о цветах
            self.color_info = color_info
            
            # Преобразуем результаты в формат BGR для обратной совместимости
            top_colors = color_info.get("top_colors", [])
            avg_bgr = color_info.get("bgr_avg", [0, 0, 0])
            
            if top_colors:
                # Сохраняем средний BGR для каждого доминирующего цвета
                # Используем средний BGR из всего ROI (можно улучшить, вычисляя для каждого цвета отдельно)
                self.dominant_colors = [tuple(avg_bgr)] * min(len(top_colors), top_n)
            else:
                # Если цвета не определены, используем средний цвет ROI
                if not avg_bgr or sum(avg_bgr) == 0:
                    avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                    self.dominant_colors = [tuple(map(int, avg_color))]
                else:
                    self.dominant_colors = [tuple(avg_bgr)]
        except Exception as e:
            # Fallback на простой метод при ошибке
            avg_color = np.mean(roi.reshape(-1, 3), axis=0)
            self.dominant_colors = [tuple(map(int, avg_color))]
            self.color_info = None
    
    def update_status(self):
        """
        Обновляет статус объекта на основе истории перемещений
        """
        if len(self.position_history) < 2:
            self.status = ObjectStatus.UNKNOWN
            return
        
        # Вычисляем среднее перемещение за последние кадры
        movements = []
        for i in range(1, min(len(self.position_history), 10)):  # Последние 10 позиций
            prev_x1, prev_y1, prev_x2, prev_y2 = self.position_history[-i-1]
            curr_x1, curr_y1, curr_x2, curr_y2 = self.position_history[-i]
            
            # Центр предыдущей позиции
            prev_center_x = (prev_x1 + prev_x2) / 2
            prev_center_y = (prev_y1 + prev_y2) / 2
            
            # Центр текущей позиции
            curr_center_x = (curr_x1 + curr_x2) / 2
            curr_center_y = (curr_y1 + curr_y2) / 2
            
            # Расстояние перемещения
            distance = np.sqrt((curr_center_x - prev_center_x)**2 + 
                             (curr_center_y - prev_center_y)**2)
            movements.append(distance)
        
        if not movements:
            self.status = ObjectStatus.UNKNOWN
            return
        
        avg_movement = np.mean(movements)
        
        # Определяем статус
        if avg_movement < self.stay_threshold:
            self.status = ObjectStatus.STAY
        elif avg_movement >= self.movement_threshold:
            if self.object_type == 'train':
                self.status = ObjectStatus.WORK  # Поезд в движении = работает
            else:
                self.status = ObjectStatus.GO
        else:
            # Среднее перемещение - может быть медленное движение
            if self.object_type == 'train':
                self.status = ObjectStatus.WORK
            else:
                self.status = ObjectStatus.GO
    
    def get_info_dict(self) -> Dict:
        """
        Возвращает словарь с информацией об объекте
        
        Returns:
            словарь с информацией
        """
        info = {
            'object_id': self.object_id,
            'object_type': self.object_type,
            'class_id': self.class_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'frame_count': self.frame_count,
            'status': self.status.value,
            'dominant_colors': self.dominant_colors,
            'train_number': self.train_number,
            'first_seen_frame': self.frame_num,
            'last_update_frame': self.last_update_frame
        }
        
        # Добавляем полную информацию о цветах, если доступна
        if self.color_info:
            info['color_info'] = {
                'top_colors': self.color_info.get('top_colors', []),
                'all_percentages': self.color_info.get('all_percentages', {}),
                'bgr_avg': self.color_info.get('bgr_avg', [])
            }
        
        return info
    
    def __repr__(self):
        return (f"ScreenObject(id={self.object_id}, type={self.object_type}, "
                f"status={self.status.value}, frames={self.frame_count})")


"""
Модуль для представления объектов на экране
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import cv2
from src.image_utils import extract_roi


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
        """
        Инициализация объекта на экране
        
        Args:
            object_id: уникальный ID объекта в группе по типу
            object_type: тип объекта ('person', 'train', и т.д.)
            class_id: ID класса объекта (0 - person, 6 - train)
            bbox: координаты (x1, y1, x2, y2)
            confidence: уверенность детекции
            frame_num: номер кадра создания
            features: вектор признаков для re-identification
        """
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
        self.dominant_colors: List[Tuple[int, int, int]] = []  # BGR формат
        
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
    
    def update_colors(self, frame: np.ndarray):
        """
        Обновляет основные цвета объекта на основе текущего кадра
        
        Args:
            frame: кадр изображения (BGR)
        """
        roi = extract_roi(frame, *self.bbox)
        if roi is None or roi.size == 0:
            return
        
        # Используем K-means для определения основных цветов
        # Упрощенный вариант - берем средние цвета по квадрантам
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            # Если ROI слишком маленький, берем средний цвет
            avg_color = np.mean(roi.reshape(-1, 3), axis=0)
            self.dominant_colors = [tuple(map(int, avg_color))]
        else:
            # Делим на квадранты и берем средний цвет каждого
            colors = []
            quadrants = [
                (0, 0, w//2, h//2),           # Верхний левый
                (w//2, 0, w, h//2),           # Верхний правый
                (0, h//2, w//2, h),          # Нижний левый
                (w//2, h//2, w, h)           # Нижний правый
            ]
            
            for x1, y1, x2, y2 in quadrants:
                quadrant = roi[y1:y2, x1:x2]
                if quadrant.size > 0:
                    avg_color = np.mean(quadrant.reshape(-1, 3), axis=0)
                    colors.append(tuple(map(int, avg_color)))
            
            # Обновляем основные цвета (берем до 4 самых частых)
            if colors:
                # Если цветов много, берем наиболее отличающиеся
                if len(colors) > 4:
                    # Упрощенный алгоритм - берем первые 4
                    self.dominant_colors = colors[:4]
                else:
                    self.dominant_colors = colors
    
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
        return {
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
    
    def __repr__(self):
        return (f"ScreenObject(id={self.object_id}, type={self.object_type}, "
                f"status={self.status.value}, frames={self.frame_count})")


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
        
        # Счетчики кадров по состояниям
        self.stay_frames = 0   # Количество кадров в состоянии STAY
        self.go_frames = 0      # Количество кадров в состоянии GO
        self.work_frames = 0   # Количество кадров в состоянии WORK
        
        # Дополнительная информация
        self.train_number: Optional[str] = None  # Номер поезда (для train)
        # Профессия определяется автоматически на основе цветов
        # Для поездов: "поезд", для людей: по умолчанию "работник"
        self.profession: Optional[str] = "поезд" if object_type == "train" else "работник"
        # Атрибуты (PPE и одежда) для людей
        self.attributes: Dict[str, List[str]] = {"ppe": [], "clothes": []}
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
            
            # Определяем профессию на основе цветов
            self._determine_profession()
        except Exception as e:
            # Fallback на простой метод при ошибке
            avg_color = np.mean(roi.reshape(-1, 3), axis=0)
            self.dominant_colors = [tuple(map(int, avg_color))]
            self.color_info = None
    
    def _determine_profession(self):
        """
        Определяет профессию объекта на основе цветов одежды
        Правила:
<<<<<<< HEAD
        - Для поездов: профессия всегда "поезд" (фильтрация по цветам происходит в filters.py)
        - Белый или серый >20% -> рабочий (приоритет выше красного)
        - Красный цвет -> инженер
        - Остальные -> работник
        """
        # Для поездов профессия всегда "поезд"
        # Фильтрация объектов с неприемлемыми цветами (голубой > 5%) происходит в filters.py
=======
        - Для поездов: профессия всегда "поезд"
        - Белый или серый >20% -> рабочий (приоритет выше красного)
        - Красный цвет -> инженер
        - Остальные -> главный инженер
        """
        # Для поездов профессия всегда "поезд"
>>>>>>> ccd72d230b67a6c5fba953dc2a0748b190518a89
        if self.object_type == "train":
            self.profession = "поезд"
            return
        
        if not self.color_info:
            self.profession = "работник"  # По умолчанию
            return
        
        all_percentages = self.color_info.get("all_percentages", {})
        top_colors = self.color_info.get("top_colors", [])
        
        # СНАЧАЛА проверяем белый и серый цвета (>20%) - приоритет выше красного
        white_percentage = all_percentages.get("white", 0.0)
        gray_percentage = all_percentages.get("gray", 0.0)
        light_gray_percentage = all_percentages.get("light_gray", 0.0)
        dark_gray_percentage = all_percentages.get("dark_gray", 0.0)
        
        total_gray_white = white_percentage + gray_percentage + light_gray_percentage + dark_gray_percentage
        
        if total_gray_white > 0.2:  # 20%
            self.profession = "рабочий"
            return
        
        # Затем проверяем наличие красного цвета
        red_percentage = all_percentages.get("red", 0.0)
        # Также проверяем в top_colors
        for color_item in top_colors:
            if color_item.get("name") == "red":
                red_percentage = max(red_percentage, color_item.get("percentage", 0.0))
        
        if red_percentage > 0.0:
            self.profession = "инженер"
            return
        
        # Остальные случаи
        self.profession = "работник"
    
    def update_status(self, frame_width: Optional[int] = None):
        """
        Обновляет статус объекта на основе истории перемещений и положения на экране
        
        Args:
            frame_width: ширина кадра для определения середины экрана (для статуса WORK)
        """
        # Получаем текущую позицию
        curr_x1, curr_y1, curr_x2, curr_y2 = self.bbox
        curr_center_x = (curr_x1 + curr_x2) / 2
        
        # Проверяем положение относительно середины экрана (для статуса WORK)
        is_on_right_side = False
        if frame_width is not None:
            screen_middle = frame_width / 2
            is_on_right_side = curr_center_x > screen_middle
        
        # Если только одна позиция в истории, определяем статус только по положению
        if len(self.position_history) < 2:
            if is_on_right_side:
                self.status = ObjectStatus.WORK
                self.work_frames += 1
            else:
                self.status = ObjectStatus.UNKNOWN
            return
        
        # Получаем текущую позицию
        curr_x1, curr_y1, curr_x2, curr_y2 = self.bbox
        curr_center_x = (curr_x1 + curr_x2) / 2
        
        # Проверяем положение относительно середины экрана (для статуса WORK)
        is_on_right_side = False
        if frame_width is not None:
            screen_middle = frame_width / 2
            is_on_right_side = curr_center_x > screen_middle
        
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
        
        # Сохраняем предыдущий статус для обновления счетчиков
        prev_status = self.status
        
        # Определяем статус
        # WORK определяется по положению справа от середины экрана
        if is_on_right_side:
            self.status = ObjectStatus.WORK
        elif avg_movement < self.stay_threshold:
            self.status = ObjectStatus.STAY
        elif avg_movement >= self.movement_threshold:
            self.status = ObjectStatus.GO
        else:
            # Среднее перемещение - медленное движение
            self.status = ObjectStatus.GO
        
        # Обновляем счетчики кадров (увеличиваем каждый кадр)
        if self.status == ObjectStatus.STAY:
            self.stay_frames += 1
        elif self.status == ObjectStatus.GO:
            self.go_frames += 1
        elif self.status == ObjectStatus.WORK:
            self.work_frames += 1
    
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
            'stay_frames': self.stay_frames,
            'go_frames': self.go_frames,
            'work_frames': self.work_frames,
            'profession': self.profession,
            'attributes': self.attributes,
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


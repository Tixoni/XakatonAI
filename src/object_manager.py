"""
Модуль для управления объектами на экране
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from src.screen_object import ScreenObject, ObjectStatus
from src.tracker import ReIDTracker, Track
import numpy as np


class ObjectManager:
    """Менеджер для управления объектами на экране"""
    
    def __init__(self, class_names: Dict[int, str]):
        """
        Инициализация менеджера объектов
        
        Args:
            class_names: словарь соответствия class_id -> имя класса
        """
        self.class_names = class_names
        # Хранилище объектов по типу: {object_type: {object_id: ScreenObject}}
        self.objects_by_type: Dict[str, Dict[int, ScreenObject]] = defaultdict(dict)
        # Счетчики ID для каждого типа объекта
        self.next_id_by_type: Dict[str, int] = defaultdict(int)
        # Связь track_id -> object_id для каждого типа
        self.track_to_object: Dict[int, Tuple[str, int]] = {}  # track_id -> (object_type, object_id)
    
    def get_object_type(self, class_id: int) -> str:
        
        return self.class_names.get(class_id, f"class_{class_id}").lower()
    
    def create_object_from_track(self, track: Track, frame_num: int) -> ScreenObject:
        """
        Создает ScreenObject из Track
        
        Args:
            track: трек объекта
            frame_num: номер кадра
            
        Returns:
            созданный ScreenObject
        """
        object_type = self.get_object_type(track.class_id)
        
        # Получаем следующий ID для этого типа
        object_id = self.next_id_by_type[object_type]
        self.next_id_by_type[object_type] += 1
        
        # Создаем объект
        screen_object = ScreenObject(
            object_id=object_id,
            object_type=object_type,
            class_id=track.class_id,
            bbox=track.bbox,
            confidence=track.confidence,
            frame_num=frame_num,
            features=track.features
        )
        
        # Копируем дополнительную информацию
        if hasattr(track, 'train_number'):
            screen_object.train_number = track.train_number
        
        # Сохраняем объект
        self.objects_by_type[object_type][object_id] = screen_object
        
        # Сохраняем связь track_id -> object_id
        self.track_to_object[track.track_id] = (object_type, object_id)
        
        return screen_object
    
    def update_object_from_track(self, track: Track, frame_num: int, frame: np.ndarray) -> Optional[ScreenObject]:
        """
        Обновляет существующий объект из трека или создает новый
        
        Args:
            track: трек объекта
            frame_num: номер кадра
            frame: кадр изображения для обновления цветов
            
        Returns:
            обновленный или созданный ScreenObject
        """
        # Проверяем, есть ли уже объект для этого трека
        if track.track_id in self.track_to_object:
            object_type, object_id = self.track_to_object[track.track_id]
            screen_object = self.objects_by_type[object_type].get(object_id)
            
            if screen_object:
                # Обновляем существующий объект
                screen_object.update(
                    bbox=track.bbox,
                    confidence=track.confidence,
                    frame_num=frame_num,
                    features=track.features
                )
                
                # Обновляем цвета (каждый N-й кадр для оптимизации)
                if frame_num % 5 == 0:  # Обновляем цвета каждые 5 кадров
                    # Используем компенсацию освещения для лучшего определения цветов
                    lighting_compensation = {"enabled": True, "normalize_brightness": False, "wider_color_ranges": True}
                    screen_object.update_colors(frame, top_n=4, lighting_compensation=lighting_compensation)
                
                # Обновляем статус
                screen_object.update_status()
                
                # Обновляем номер поезда, если есть
                if hasattr(track, 'train_number') and track.train_number:
                    screen_object.train_number = track.train_number
                
                return screen_object
        
        # Если объекта нет, создаем новый
        return self.create_object_from_track(track, frame_num)
    
    def get_object_by_track_id(self, track_id: int) -> Optional[ScreenObject]:
        """
        Получает объект по track_id
        
        Args:
            track_id: ID трека
            
        Returns:
            ScreenObject или None
        """
        if track_id not in self.track_to_object:
            return None
        
        object_type, object_id = self.track_to_object[track_id]
        return self.objects_by_type[object_type].get(object_id)
    
    def get_objects_by_type(self, object_type: str) -> List[ScreenObject]:
        """
        Получает все объекты определенного типа
        
        Args:
            object_type: тип объекта
            
        Returns:
            список объектов
        """
        return list(self.objects_by_type[object_type].values())
    
    def get_all_objects(self) -> List[ScreenObject]:
        """
        Получает все объекты
        
        Returns:
            список всех объектов
        """
        all_objects = []
        for objects_dict in self.objects_by_type.values():
            all_objects.extend(objects_dict.values())
        return all_objects
    
    def remove_inactive_objects(self, active_track_ids: set, max_age: int = 150):
        """
        Удаляет неактивные объекты (треки которых больше не существуют)
        
        Args:
            active_track_ids: множество активных track_id
            max_age: максимальный возраст объекта без обновления
        """
        # Находим неактивные треки
        inactive_tracks = set(self.track_to_object.keys()) - active_track_ids
        
        for track_id in list(inactive_tracks):
            if track_id in self.track_to_object:
                object_type, object_id = self.track_to_object[track_id]
                screen_object = self.objects_by_type[object_type].get(object_id)
                
                if screen_object:
                    # Проверяем возраст объекта
                    frames_since_update = screen_object.last_update_frame - screen_object.frame_num
                    if frames_since_update > max_age:
                        # Удаляем объект
                        del self.objects_by_type[object_type][object_id]
                        del self.track_to_object[track_id]
    
    def get_statistics(self) -> Dict:
        """
        Получает статистику по объектам
        
        Returns:
            словарь со статистикой
        """
        stats = {}
        for object_type, objects_dict in self.objects_by_type.items():
            stats[object_type] = {
                'total': len(objects_dict),
                'by_status': {},
                'by_colors': {}  # Статистика по цветам
            }
            
            # Подсчет по статусам и цветам
            for obj in objects_dict.values():
                # Статусы
                status = obj.status.value
                if status not in stats[object_type]['by_status']:
                    stats[object_type]['by_status'][status] = 0
                stats[object_type]['by_status'][status] += 1
                
                # Цвета
                if obj.color_info and obj.color_info.get('top_colors'):
                    # Берем основной цвет (первый в списке)
                    top_colors = obj.color_info.get('top_colors', [])
                    if top_colors:
                        main_color = top_colors[0].get('name', 'unknown')
                        if main_color not in stats[object_type]['by_colors']:
                            stats[object_type]['by_colors'][main_color] = 0
                        stats[object_type]['by_colors'][main_color] += 1
                else:
                    # Если цвет не определен
                    if 'unknown' not in stats[object_type]['by_colors']:
                        stats[object_type]['by_colors']['unknown'] = 0
                    stats[object_type]['by_colors']['unknown'] += 1
        
        return stats


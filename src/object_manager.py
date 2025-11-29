"""
Модуль для управления объектами на экране
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from src.screen_object import ScreenObject, ObjectStatus
from src.tracker import ReIDTracker, Track
from src.image_utils import extract_roi
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
    
    def update_object_from_track(self, track: Track, frame_num: int, frame: np.ndarray,
                                 attribute_models=None, attr_update_interval=10) -> Optional[ScreenObject]:
        """
        Обновляет существующий объект из трека или создает новый
        
        Args:
            track: трек объекта
            frame_num: номер кадра
            frame: кадр изображения для обновления цветов
            attribute_models: модели для детекции атрибутов (PPE и одежда)
            attr_update_interval: интервал обновления атрибутов (в кадрах)
            
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
                
                # Обновляем атрибуты (PPE и одежда) для людей
                if attribute_models and attribute_models.is_enabled() and object_type == "person":
                    # Обновляем атрибуты периодически
                    if frame_num % attr_update_interval == 0:
                        x1, y1, x2, y2 = track.bbox
                        person_crop = extract_roi(frame, x1, y1, x2, y2, crop_border_ratio=0.0)
                        if person_crop is not None and person_crop.size > 0:
                            try:
                                from src.attribute_models import map_ppe_names, map_clothes_names
                                ppe = attribute_models.run_ppe(person_crop)
                                clothes = attribute_models.run_clothes(person_crop)
                                ppe_names = map_ppe_names(ppe)
                                clothes_names = map_clothes_names(clothes)
                                # Объединяем с существующими атрибутами (уникальные значения)
                                existing_ppe = screen_object.attributes.get("ppe", [])
                                existing_clothes = screen_object.attributes.get("clothes", [])
                                screen_object.attributes = {
                                    "ppe": sorted(set(existing_ppe + ppe_names)),
                                    "clothes": sorted(set(existing_clothes + clothes_names))
                                }
                            except Exception as e:
                                pass  # Игнорируем ошибки при обновлении атрибутов
                
                # Обновляем статус (передаем ширину кадра для определения WORK)
                frame_height, frame_width = frame.shape[:2]
                screen_object.update_status(frame_width=frame_width)
                
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
    
    def get_statistics(self, min_frames: int = 20) -> Dict:
        """
        Получает статистику по объектам
        
        Args:
            min_frames: минимальное количество кадров для учета объекта в статистике
        
        Returns:
            словарь со статистикой
        """
        stats = {}
        for object_type, objects_dict in self.objects_by_type.items():
            # Фильтруем объекты по минимальному количеству кадров
            valid_objects = [obj for obj in objects_dict.values() if obj.frame_count >= min_frames]
            
            stats[object_type] = {
                'total': len(valid_objects),
                'by_status': {},
                'by_colors': {},  # Статистика по цветам
                'by_profession': {},  # Статистика по профессиям
                'by_ppe': {},  # Статистика по PPE
                'by_clothes': {}  # Статистика по одежде
            }
            
            # Подсчет по статусам, цветам и профессиям
            for obj in valid_objects:
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
                
                # Профессии (только для людей)
                if obj.profession:
                    profession = obj.profession
                    if profession not in stats[object_type]['by_profession']:
                        stats[object_type]['by_profession'][profession] = 0
                    stats[object_type]['by_profession'][profession] += 1
                
                # Атрибуты (PPE и одежда) - только для людей
                if obj.attributes and object_type == "person":
                    ppe_list = obj.attributes.get("ppe", [])
                    clothes_list = obj.attributes.get("clothes", [])
                    
                    # Статистика по PPE
                    for ppe_item in ppe_list:
                        if ppe_item not in stats[object_type]['by_ppe']:
                            stats[object_type]['by_ppe'][ppe_item] = 0
                        stats[object_type]['by_ppe'][ppe_item] += 1
                    
                    # Статистика по одежде
                    for clothes_item in clothes_list:
                        if clothes_item not in stats[object_type]['by_clothes']:
                            stats[object_type]['by_clothes'][clothes_item] = 0
                        stats[object_type]['by_clothes'][clothes_item] += 1
        
        return stats


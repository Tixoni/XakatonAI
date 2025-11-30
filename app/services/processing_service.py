"""
Сервис для обработки изображений и видео
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Добавляем путь к src модулям
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.object_manager import ObjectManager
from src.filters import apply_color_filters
from src.image_utils import resize_frame
from src.db import insert_worker, insert_train
from src.uploader import upload_file
from app.models import ObjectInfo, Statistics, DetectionResponse
from app.services.detector_service import DetectorService


class ProcessingService:
    """Сервис для обработки изображений и видео"""
    
    def __init__(self, detector_service: DetectorService, config: Dict[str, Any]):
        self.detector_service = detector_service
        self.config = config
        self.object_manager = ObjectManager(detector_service.detector.CLASSES)
    
    def process_image(
        self, 
        image: np.ndarray,
        request_params: Dict[str, Any]
    ) -> Tuple[List[ObjectInfo], Optional[np.ndarray], Dict[str, Any]]:
        """
        Обработка изображения
        
        Args:
            image: изображение (numpy array)
            request_params: параметры запроса
            
        Returns:
            tuple: (список объектов, обработанное изображение, статистика)
        """
        start_time = time.time()
        
        # Получаем параметры
        confidence_threshold = request_params.get("confidence_threshold", 0.5)
        target_classes = request_params.get("target_classes", [0, 6])
        return_image = request_params.get("return_image", True)
        
        # Обновляем порог уверенности детектора
        if hasattr(self.detector_service.detector, 'conf_threshold'):
            self.detector_service.detector.conf_threshold = confidence_threshold
        
        # Детекция
        detections = self.detector_service.detector.detect(image, target_classes=target_classes)
        
        # Применяем цветовые фильтры
        color_filters = self.config.get("color_filters", {})
        debug_cfg = self.config.get("debug", {})
        detections, rejected = apply_color_filters(
            image, detections, color_filters, 
            self.detector_service.detector.CLASSES, debug_cfg
        )
        
        # Обработка через трекер (для единообразия с видео)
        result_image = None
        if return_image:
            result_image = image.copy()
            result_image = self.detector_service.detector.draw_detections(result_image, detections)
        
        # Создаем объекты из детекций
        objects = []
        for i, (class_id, conf, x1, y1, x2, y2) in enumerate(detections):
            object_type = self.object_manager.get_object_type(class_id)
            object_id = self.object_manager.next_id_by_type[object_type]
            self.object_manager.next_id_by_type[object_type] += 1
            
            # Создаем ScreenObject
            from src.screen_object import ScreenObject
            screen_object = ScreenObject(
                object_id=object_id,
                object_type=object_type,
                class_id=class_id,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                frame_num=0
            )
            
            # Обновляем цвета
            screen_object.update_colors(image, top_n=4)
            
            # Обновляем статус
            frame_height, frame_width = image.shape[:2]
            screen_object.update_status(frame_width=frame_width)
            
            # Обновляем атрибуты для людей
            if (self.detector_service.attribute_models and 
                self.detector_service.attribute_models.is_enabled() and 
                object_type == "person"):
                from src.image_utils import extract_roi
                from src.attribute_models import map_ppe_names, map_clothes_names
                
                person_crop = extract_roi(image, x1, y1, x2, y2, crop_border_ratio=0.0)
                if person_crop is not None and person_crop.size > 0:
                    try:
                        ppe = self.detector_service.attribute_models.run_ppe(person_crop)
                        clothes = self.detector_service.attribute_models.run_clothes(person_crop)
                        ppe_names = map_ppe_names(ppe)
                        clothes_names = map_clothes_names(clothes)
                        screen_object.attributes = {
                            "ppe": sorted(set(ppe_names)),
                            "clothes": sorted(set(clothes_names))
                        }
                    except Exception:
                        pass
            
            # Сохраняем объект
            self.object_manager.objects_by_type[object_type][object_id] = screen_object
            
            # Преобразуем в ObjectInfo
            obj_info = screen_object.get_info_dict()
            # Преобразуем bbox из кортежа в список
            if isinstance(obj_info.get('bbox'), tuple):
                obj_info['bbox'] = list(obj_info['bbox'])
            # Убеждаемся, что dominant_colors - список списков
            if obj_info.get('dominant_colors'):
                obj_info['dominant_colors'] = [
                    list(color) if isinstance(color, tuple) else color
                    for color in obj_info['dominant_colors']
                ]
            objects.append(ObjectInfo(**obj_info))
        
        # Статистика
        stats = self.object_manager.get_statistics(min_frames=1)
        
        # Убеждаемся, что статистика имеет правильную структуру
        for obj_type, stat_data in stats.items():
            # Убеждаемся, что все обязательные поля присутствуют
            if 'by_ppe' not in stat_data:
                stat_data['by_ppe'] = {}
            if 'by_clothes' not in stat_data:
                stat_data['by_clothes'] = {}
        
        processing_time = time.time() - start_time
        
        return objects, result_image, {
            "statistics": stats,
            "processing_time": processing_time,
            "total_detections": len(detections),
            "rejected": len(rejected)
        }
    
    def process_video(
        self,
        video_path: str,
        request_params: Dict[str, Any]
    ) -> Tuple[List[ObjectInfo], Optional[str], Dict[str, Any]]:
        """
        Обработка видео
        
        Args:
            video_path: путь к видео файлу
            request_params: параметры запроса
            
        Returns:
            tuple: (список объектов, путь к обработанному видео, метаданные)
        """
        start_time = time.time()
        
        # Обновляем конфиг с параметрами запроса
        config = self.config.copy()
        if "confidence_threshold" in request_params:
            if "detection" not in config:
                config["detection"] = {}
            config["detection"]["confidence_threshold"] = request_params["confidence_threshold"]
        if "target_classes" in request_params:
            if "detection" not in config:
                config["detection"] = {}
            config["detection"]["target_classes"] = request_params["target_classes"]
        
        # Обновляем порог уверенности детектора
        if hasattr(self.detector_service.detector, 'conf_threshold'):
            self.detector_service.detector.conf_threshold = request_params.get("confidence_threshold", 0.5)
        
        # Создаем новый ObjectManager для этого видео
        video_object_manager = ObjectManager(self.detector_service.detector.CLASSES)
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Параметры оптимизации
        opt_config = config.get("video_optimization", {})
        frame_skip = opt_config.get("frame_skip", 1)
        max_width = opt_config.get("max_width", 1280)
        max_height = opt_config.get("max_height", 540)
        
        # Обработка кадров
        frame_count = 0
        processed_count = 0
        target_classes = request_params.get("target_classes", [0, 6])
        return_video = request_params.get("return_video", True)
        
        output_dir = Path(config.get("processing", {}).get("output_dir", "results"))
        output_dir.mkdir(exist_ok=True)
        
        # Создаем writer для выходного видео (если нужно)
        output_path = None
        writer = None
        if return_video:
            output_filename = f"processed_{Path(video_path).stem}_{int(time.time())}.mp4"
            output_path = str(output_dir / output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Получаем размер первого кадра для writer
            ret, first_frame = cap.read()
            if ret:
                if first_frame.shape[1] > max_width or first_frame.shape[0] > max_height:
                    first_frame, _ = resize_frame(first_frame, max_width, max_height, True, False)
                frame_width = first_frame.shape[1]
                frame_height = first_frame.shape[0]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Возвращаемся к началу
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Пропускаем кадры
                if frame_count % frame_skip != 0:
                    continue
                
                processed_count += 1
                
                # Изменяем размер если нужно
                resized_frame = frame
                if frame.shape[1] > max_width or frame.shape[0] > max_height:
                    resized_frame, _ = resize_frame(frame, max_width, max_height, True, False)
                
                # Детекция
                detections = self.detector_service.detector.detect(resized_frame, target_classes=target_classes)
                
                # Применяем фильтры
                color_filters = config.get("color_filters", {})
                debug_cfg = config.get("debug", {})
                detections, rejected = apply_color_filters(
                    resized_frame, detections, color_filters,
                    self.detector_service.detector.CLASSES, debug_cfg
                )
                
                # Обработка через трекер
                tracks = None
                if self.detector_service.tracker:
                    tracks = self.detector_service.tracker.update(resized_frame, detections)
                    
                    # Обновляем объекты
                    active_track_ids = set()
                    for track_data in tracks:
                        if len(track_data) >= 7:
                            track_id, class_id, conf, x1, y1, x2, y2 = track_data[:7]
                            active_track_ids.add(track_id)
                            
                            # Получаем Track объект из трекера
                            track_obj = self.detector_service.tracker.tracks.get(track_id)
                            if track_obj:
                                video_object_manager.update_object_from_track(
                                    track_obj, processed_count, resized_frame,
                                    self.detector_service.attribute_models,
                                    config.get("attributes", {}).get("update_interval", 10)
                                )
                    
                    # Удаляем неактивные объекты
                    video_object_manager.remove_inactive_objects(
                        active_track_ids,
                        max_age=config.get("re_identification", {}).get("max_age", 150)
                    )
                    
                    # Распознавание номеров поездов через OCR
                    ocr_cfg = config.get("train_number_ocr", {})
                    if ocr_cfg.get("enabled", False) and self.detector_service.ocr_reader:
                        ocr_frame_skip = ocr_cfg.get("frame_skip", 10)  # Проверять каждый N-й кадр
                        use_bottom_right_quadrant = ocr_cfg.get("use_bottom_right_quadrant", True)
                        
                        # Проверяем только каждый N-й кадр для оптимизации
                        if processed_count % ocr_frame_skip == 0:
                            train_number = None
                            
                            # Используем правую половину экрана (оптимизированный вариант)
                            if use_bottom_right_quadrant:
                                try:
                                    train_number = self.detector_service.ocr_reader.recognize_from_right_half(resized_frame)
                                    if train_number:
                                        # Присваиваем номер всем поездам в кадре
                                        if tracks:
                                            for track_data in tracks:
                                                if len(track_data) >= 7:
                                                    track_id, class_id = track_data[0], track_data[1]
                                                    if class_id == 6:  # train
                                                        track_obj = self.detector_service.tracker.tracks.get(track_id)
                                                        if track_obj:
                                                            track_obj.train_number = train_number
                                                            print(f"[OCR] Распознан номер поезда: {train_number} (из правого нижнего квадранта)")
                                except Exception as e:
                                    if debug_cfg.get("log_detection_details"):
                                        print(f"Ошибка OCR (правый нижний квадрант): {e}")
                            else:
                                # Используем область детектированного поезда
                                if tracks:
                                    for track_data in tracks:
                                        if len(track_data) >= 7:
                                            track_id, class_id, conf, x1, y1, x2, y2 = track_data[:7]
                                            if class_id == 6:  # train
                                                track_obj = self.detector_service.tracker.tracks.get(track_id)
                                                if track_obj and track_obj.train_number is None:
                                                    try:
                                                        roi_offset = ocr_cfg.get("roi_offset", None)
                                                        train_number = self.detector_service.ocr_reader.recognize_from_train_bbox(
                                                            resized_frame, (x1, y1, x2, y2), roi_offset
                                                        )
                                                        if train_number:
                                                            track_obj.train_number = train_number
                                                            print(f"[OCR] Распознан номер поезда ID:{track_id}: {train_number}")
                                                    except Exception as e:
                                                        if debug_cfg.get("log_detection_details"):
                                                            print(f"Ошибка OCR для трека {track_id}: {e}")
                                
                                # Также пробуем фиксированную область, если настроена
                                fixed_roi = ocr_cfg.get("fixed_roi", None)
                                if fixed_roi and fixed_roi.get("enabled", False):
                                    try:
                                        train_number = self.detector_service.ocr_reader.recognize_train_number(
                                            resized_frame, fixed_roi
                                        )
                                        if train_number:
                                            # Присваиваем номер всем поездам в кадре
                                            if tracks:
                                                for track_data in tracks:
                                                    if len(track_data) >= 7:
                                                        track_id, class_id = track_data[0], track_data[1]
                                                        if class_id == 6:  # train
                                                            track_obj = self.detector_service.tracker.tracks.get(track_id)
                                                            if track_obj:
                                                                track_obj.train_number = train_number
                                                                if processed_count % 30 == 0:
                                                                    print(f"[OCR] Распознан номер поезда: {train_number} (из фиксированной области)")
                                    except Exception as e:
                                        if debug_cfg.get("log_detection_details"):
                                            print(f"Ошибка OCR (фиксированная область): {e}")
                
                # Отрисовка для видео
                if writer:
                    if tracks and len(tracks) > 0:
                        # Отрисовка с треками
                        result_frame = self.detector_service.detector.draw_detections(
                            resized_frame, tracks, show_track_ids=True
                        )
                    else:
                        # Отрисовка без треков
                        result_frame = self.detector_service.detector.draw_detections(
                            resized_frame, detections, show_track_ids=False
                        )
                    writer.write(result_frame)
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Получаем все объекты
        all_objects = video_object_manager.get_all_objects()
        objects = []
        for obj in all_objects:
            obj_info = obj.get_info_dict()
            # Преобразуем bbox из кортежа в список
            if isinstance(obj_info.get('bbox'), tuple):
                obj_info['bbox'] = list(obj_info['bbox'])
            # Убеждаемся, что dominant_colors - список списков
            if obj_info.get('dominant_colors'):
                obj_info['dominant_colors'] = [
                    list(color) if isinstance(color, tuple) else color
                    for color in obj_info['dominant_colors']
                ]
            objects.append(ObjectInfo(**obj_info))
        
        # Статистика
        stats = video_object_manager.get_statistics(min_frames=1)
        
        # Убеждаемся, что статистика имеет правильную структуру
        for obj_type, stat_data in stats.items():
            # Убеждаемся, что все обязательные поля присутствуют
            if 'by_ppe' not in stat_data:
                stat_data['by_ppe'] = {}
            if 'by_clothes' not in stat_data:
                stat_data['by_clothes'] = {}
        
        processing_time = time.time() - start_time
        
        # Сохранение данных в БД (всегда, независимо от return_video)
        print(f"\n=== Сохранение данных в БД ===")
        try:
            # Проверяем подключение к БД
            from src.db import get_conn
            test_conn = get_conn()
            test_conn.close()
            print("Подключение к БД успешно")
            
            # Фильтруем объекты с минимальным количеством кадров (>= 20)
            min_frames_threshold = 20
            valid_objects = [obj for obj in all_objects if obj.frame_count >= min_frames_threshold]
            
            print(f"Найдено объектов для сохранения: {len(valid_objects)}")
            
            if len(valid_objects) == 0:
                print("Нет объектов для сохранения (нужно >= 20 кадров)")
            else:
                # Создаем обратный словарь для быстрого поиска track_id
                object_to_track = {}
                print(f"Всего связей track_to_object: {len(video_object_manager.track_to_object)}")
                for track_id, (obj_type, obj_id) in video_object_manager.track_to_object.items():
                    object_to_track[(obj_type, obj_id)] = track_id
                    if obj_type == "train":
                        print(f"  Связь поезда: track_id={track_id} -> (type={obj_type}, object_id={obj_id})")
                
                # Сохраняем данные о людях
                people_count = 0
                people_errors = 0
                for obj in valid_objects:
                    if obj.object_type == "person":
                        # Получаем цвета объекта
                        color = None
                        if obj.color_info and obj.color_info.get('top_colors'):
                            top_colors = obj.color_info.get('top_colors', [])
                            color_names = [color_item.get('name', 'unknown') for color_item in top_colors[:3]]
                            color = ",".join(color_names) if color_names else None
                        
                        # Получаем track_id из обратного словаря
                        track_id = object_to_track.get((obj.object_type, obj.object_id))
                        
                        if track_id is not None:
                            try:
                                insert_worker(
                                    track_id=track_id,
                                    color=color,
                                    stand_frames=obj.stay_frames,
                                    walk_frames=obj.go_frames,
                                    work_frames=obj.work_frames,
                                    attributes=obj.attributes if obj.attributes else None
                                )
                                people_count += 1
                                print(f"✓ Работник track_id={track_id} сохранен в БД")
                            except Exception as e:
                                people_errors += 1
                                error_msg = str(e)
                                print(f"✗ Ошибка при сохранении работника track_id={track_id}: {error_msg}")
                                if people_errors <= 3 and ("Connection refused" in error_msg or "connection to server" in error_msg.lower()):
                                    print("БД недоступна. Прерываем сохранение.")
                                    break
                        else:
                            print(f"Не найден track_id для работника object_id={obj.object_id}")
                
                # Сохраняем данные о поездах в таблицу trains
                trains_count = 0
                trains_errors = 0
                train_objects = [obj for obj in valid_objects if obj.object_type == "train"]
                print(f"Найдено поездов для сохранения: {len(train_objects)}")
                
                # Отладочная информация
                if len(train_objects) > 0:
                    print(f"Доступные связи в object_to_track для поездов:")
                    for (ot, oid), tid in object_to_track.items():
                        if ot == "train":
                            print(f"  (type={ot}, object_id={oid}) -> track_id={tid}")
                    print(f"Поезда в valid_objects:")
                    for obj in train_objects:
                        print(f"  object_id={obj.object_id}, object_type={obj.object_type}, class_id={obj.class_id}, frames={obj.frame_count}")
                
                for obj in train_objects:
                    # Получаем track_id из обратного словаря
                    track_id = object_to_track.get((obj.object_type, obj.object_id))
                    print(f"Поиск track_id для поезда: (type={obj.object_type}, object_id={obj.object_id}) -> track_id={track_id}")
                    
                    # Если track_id не найден, используем object_id как fallback (но это не идеально)
                    if track_id is None:
                        print(f"ВНИМАНИЕ: Не найден track_id для поезда object_id={obj.object_id}")
                        print(f"Попытка использовать object_id как track_id (может вызвать конфликт)")
                        # Используем отрицательный track_id для поездов без трекера
                        track_id = -obj.object_id - 1000  # Уникальный отрицательный ID
                        print(f"Используем fallback track_id={track_id}")
                    
                    try:
                        print(f"Сохранение поезда в таблицу trains: track_id={track_id}, "
                              f"номер={obj.train_number}, время={obj.frame_count} кадров")
                        insert_train(
                            track_id=track_id,
                            number=obj.train_number,
                            total_time=obj.frame_count
                        )
                        trains_count += 1
                        print(f"Поезд track_id={track_id} успешно сохранен в таблицу trains")
                    except Exception as e:
                        trains_errors += 1
                        error_msg = str(e)
                        print(f"ОШИБКА при сохранении поезда track_id={track_id} в таблицу trains: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        if trains_errors <= 3 and ("Connection refused" in error_msg or "connection to server" in error_msg.lower()):
                            print("БД недоступна. Прерываем сохранение.")
                            break
                
                print(f"Сохранено в БД: работников={people_count}, поездов={trains_count}")
                if people_errors > 0 or trains_errors > 0:
                    print(f"Ошибок при сохранении: работников={people_errors}, поездов={trains_errors}")
        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка подключения к БД: {error_msg}")
            if "Connection refused" in error_msg or "connection to server" in error_msg.lower():
                print("PostgreSQL сервер недоступен. Проверьте:")
                print("   1. PostgreSQL сервер запущен (docker-compose ps)")
                print("   2. Переменные окружения POSTGRES_* установлены")
                print("   3. Сеть между контейнерами работает")
            print("Продолжаем без сохранения в БД...")
        
        # Генерация CSV отчета
        try:
            import csv
            from datetime import datetime
            
            csv_dir = Path(config.get("processing", {}).get("output_dir", "results"))
            csv_dir.mkdir(exist_ok=True)
            
            # Имя файла с timestamp
            video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"report_{Path(video_path).stem}_{video_timestamp}.csv"
            csv_path = csv_dir / csv_filename
            
            print(f"\n=== Генерация CSV отчета ===")
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                
                # Заголовки
                writer.writerow([
                    "type",
                    "id",
                    "colors",
                    "ppe",
                    "clothes",
                    "first_seen",
                    "last_seen",
                    "train_number",
                    "arrival_time",
                    "departure_time",
                    "video_timestamp"
                ])
                
                # Получаем все объекты для CSV
                csv_objects = video_object_manager.get_all_objects()
                
                # Создаем обратный словарь для track_id
                object_to_track_csv = {}
                for track_id, (obj_type, obj_id) in video_object_manager.track_to_object.items():
                    object_to_track_csv[(obj_type, obj_id)] = track_id
                
                # Обрабатываем людей
                for obj in csv_objects:
                    if obj.object_type == "person":
                        track_id = object_to_track_csv.get((obj.object_type, obj.object_id), obj.object_id)
                        
                        # Получаем цвета
                        colors = []
                        if obj.color_info and obj.color_info.get('top_colors'):
                            colors = [color_item.get('name', 'unknown') for color_item in obj.color_info.get('top_colors', [])]
                        
                        # Получаем PPE и одежду
                        ppe_list = obj.attributes.get("ppe", []) if obj.attributes else []
                        clothes_list = obj.attributes.get("clothes", []) if obj.attributes else []
                        
                        writer.writerow([
                            "person",
                            track_id,
                            ",".join(colors),
                            ",".join(ppe_list),
                            ",".join(clothes_list),
                            obj.frame_num,  # first_seen_frame
                            obj.last_update_frame,  # last_seen_frame
                            "",  # train_number (для людей пусто)
                            "",  # arrival_time
                            "",  # departure_time
                            video_timestamp
                        ])
                
                # Обрабатываем поезда
                train_objects_csv = [obj for obj in csv_objects if obj.object_type == "train"]
                for obj in train_objects_csv:
                    track_id = object_to_track_csv.get((obj.object_type, obj.object_id), obj.object_id)
                    
                    # Для поездов arrival_time и departure_time - это first_seen и last_seen
                    writer.writerow([
                        "train",
                        track_id,
                        "",  # colors (для поездов пусто)
                        "",  # ppe
                        "",  # clothes
                        obj.frame_num,  # first_seen_frame
                        obj.last_update_frame,  # last_seen_frame
                        obj.train_number or "",  # train_number
                        obj.frame_num,  # arrival_time (first_seen)
                        obj.last_update_frame,  # departure_time (last_seen)
                        video_timestamp
                    ])
            
            print(f"[CSV] Создан: {csv_path}")
            
            # Загрузка на Yandex Cloud
            upload_file(str(csv_path), csv_filename)
            
        except Exception as e:
            print(f"[CSV ERROR] Ошибка при создании CSV: {e}")
            import traceback
            traceback.print_exc()
        
        return objects, output_path, {
            "statistics": stats,
            "processing_time": processing_time,
            "total_frames": total_frames,
            "processed_frames": processed_count
        }


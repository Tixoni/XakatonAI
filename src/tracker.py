
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from src.reid import FeatureExtractor


class Track:

    def __init__(self, track_id: int, class_id: int, bbox: Tuple[int, int, int, int], 
                 confidence: float, features: np.ndarray, frame_num: int):

        self.track_id = track_id
        self.class_id = class_id
        self.bbox = bbox
        self.confidence = confidence
        self.features = features
        self.frame_num = frame_num
        self.age = 1  # Возраст трека (количество кадров)
        self.time_since_update = 0  # Кадров с последнего обновления
        self.history = [bbox]  # История позиций
        self.train_number = None  # Номер поезда (для класса train)
        
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, 
               features: np.ndarray, frame_num: int):
        self.bbox = bbox
        self.confidence = confidence
        # Обновляем признаки как взвешенное среднее
        alpha = 0.1  # Коэффициент обновления
        self.features = (1 - alpha) * self.features + alpha * features
        self.frame_num = frame_num
        self.age += 1
        self.time_since_update = 0
        self.history.append(bbox)
        # Ограничиваем историю
        if len(self.history) > 30:
            self.history.pop(0)
    
    def predict(self):
        if len(self.history) < 2:
            return self.bbox
        
        # Простое предсказание на основе последних двух позиций
        x1, y1, x2, y2 = self.bbox
        if len(self.history) >= 2:
            prev_x1, prev_y1, prev_x2, prev_y2 = self.history[-2]
            dx1 = x1 - prev_x1
            dy1 = y1 - prev_y1
            dx2 = x2 - prev_x2
            dy2 = y2 - prev_y2
            
            pred_x1 = int(x1 + dx1)
            pred_y1 = int(y1 + dy1)
            pred_x2 = int(x2 + dx2)
            pred_y2 = int(y2 + dy2)
            
            return (pred_x1, pred_y1, pred_x2, pred_y2)
        
        return self.bbox


class ReIDTracker:
    def __init__(self, feature_extractor: FeatureExtractor, 
                 max_distance: float = 0.5,
                 max_age: int = 30,
                 min_hits: int = 1,
                 iou_threshold: float = 0.3):

        self.feature_extractor = feature_extractor
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        # Счетчик всех когда-либо созданных треков по классам
        self.all_tracks_by_class: Dict[int, set] = defaultdict(set)
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Вычисляем пересечение
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Вычисляем объединение
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _associate_detections_to_tracks(self, detections: List[Tuple], 
                                       detection_features: np.ndarray,
                                       tracks: List[Track]) -> Tuple[List[int], List[int], List[int]]:

        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Вычисляем матрицу расстояний признаков
        track_features = np.array([t.features for t in tracks])
        distance_matrix = self.feature_extractor.compute_distance_matrix(
            detection_features, track_features
        )
        
        # Вычисляем матрицу IoU
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            _, _, x1, y1, x2, y2 = det
            det_bbox = (x1, y1, x2, y2)
            for j, track in enumerate(tracks):
                iou_matrix[i, j] = self._compute_iou(det_bbox, track.bbox)
        
        # Комбинированная метрика: расстояние признаков + IoU
        # Нормализуем IoU (чем больше, тем лучше, поэтому инвертируем)
        iou_cost = 1.0 - iou_matrix
        combined_cost = 0.7 * distance_matrix + 0.3 * iou_cost
        
        # Жадное сопоставление (простой алгоритм)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Сортируем по стоимости и сопоставляем
        cost_pairs = []
        for i in range(len(detections)):
            for j in range(len(tracks)):
                cost = combined_cost[i, j]
                # Проверяем пороги
                if distance_matrix[i, j] <= self.max_distance and iou_matrix[i, j] >= self.iou_threshold:
                    cost_pairs.append((cost, i, j))
        
        cost_pairs.sort(key=lambda x: x[0])
        
        used_dets = set()
        used_tracks = set()
        
        for cost, det_idx, track_idx in cost_pairs:
            if det_idx not in used_dets and track_idx not in used_tracks:
                matched.append((det_idx, track_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def update(self, frame: np.ndarray, detections: List[Tuple]) -> List[Tuple]:

        self.frame_count += 1
        
        # Извлекаем признаки для всех детекций
        detection_features = self.feature_extractor.extract_features(frame, detections)
        
        # Получаем активные треки (не слишком старые)
        active_tracks = [t for t in self.tracks.values() 
                        if t.time_since_update < self.max_age]
        
        # Предсказываем позиции для активных треков
        for track in active_tracks:
            track.predict()
            track.time_since_update += 1
        
        # Сопоставляем детекции с треками
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
            detections, detection_features, active_tracks
        )
        
        # Обновляем сопоставленные треки
        for det_idx, track_idx in matched:
            det = detections[det_idx]
            track = active_tracks[track_idx]
            class_id, confidence, x1, y1, x2, y2 = det
            track.update((x1, y1, x2, y2), confidence, 
                        detection_features[det_idx], self.frame_count)
        
        # Создаем новые треки для несоответствующих детекций
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            class_id, confidence, x1, y1, x2, y2 = det
            track_id = self.next_id
            self.next_id += 1
            
            new_track = Track(
                track_id=track_id,
                class_id=class_id,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                features=detection_features[det_idx],
                frame_num=self.frame_count
            )
            self.tracks[track_id] = new_track
            # Сохраняем информацию о всех созданных треках
            self.all_tracks_by_class[class_id].add(track_id)
        
        # Удаляем старые треки
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update >= self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Возвращаем все треки, которые были обновлены в текущем кадре
        # (time_since_update == 0 означает, что трек был обновлен в этом кадре)
        # Также показываем новые треки сразу (age >= min_hits)
        active_tracks_list = []
        for track in self.tracks.values():
            # Показываем треки, которые были обновлены в текущем кадре (включая новые)
            # или которые достаточно старые и еще активны
            if track.time_since_update == 0:
                x1, y1, x2, y2 = track.bbox
                active_tracks_list.append((
                    track.track_id,
                    track.class_id,
                    track.confidence,
                    x1, y1, x2, y2
                ))
            elif track.age >= self.min_hits and track.time_since_update < self.max_age:
                # Показываем старые треки, которые еще активны
                x1, y1, x2, y2 = track.bbox
                active_tracks_list.append((
                    track.track_id,
                    track.class_id,
                    track.confidence,
                    x1, y1, x2, y2
                ))
        
        return active_tracks_list


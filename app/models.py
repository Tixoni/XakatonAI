"""
Pydantic модели для API запросов и ответов
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class ObjectStatus(str, Enum):
    """Статусы объектов"""
    STAY = "stay"
    GO = "go"
    WORK = "work"
    UNKNOWN = "unknown"


class DetectionRequest(BaseModel):
    """Запрос на детекцию"""
    confidence_threshold: Optional[float] = 0.5
    target_classes: Optional[List[int]] = [0, 6]  # person, train
    return_image: Optional[bool] = True
    return_statistics: Optional[bool] = True


class ObjectInfo(BaseModel):
    """Информация об объекте"""
    object_id: int
    object_type: str
    class_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    frame_count: int
    status: str
    stay_frames: int
    go_frames: int
    work_frames: int
    profession: Optional[str] = None
    attributes: Optional[Dict[str, List[str]]] = None
    dominant_colors: Optional[List[List[int]]] = None
    train_number: Optional[str] = None
    first_seen_frame: int
    last_update_frame: int


class Statistics(BaseModel):
    """Статистика по объектам"""
    total: int
    by_status: Dict[str, int]
    by_colors: Dict[str, int]
    by_profession: Dict[str, int]
    by_ppe: Optional[Dict[str, int]] = None
    by_clothes: Optional[Dict[str, int]] = None


class DetectionResponse(BaseModel):
    """Ответ на запрос детекции"""
    success: bool
    message: str
    objects: List[ObjectInfo]
    statistics: Optional[Dict[str, Statistics]] = None
    processing_time: Optional[float] = None
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    image: Optional[str] = None  # Base64 encoded image


class HealthResponse(BaseModel):
    """Ответ на проверку здоровья сервиса"""
    status: str
    version: str
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Ответ с ошибкой"""
    success: bool = False
    error: str
    detail: Optional[str] = None


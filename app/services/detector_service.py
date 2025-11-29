"""
Сервис для инициализации и управления детекторами
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Добавляем путь к src модулям
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.detector import YOLODetector
from src.ppe_detector import PPEDetector
from src.attribute_models import AttributeModels
from src.reid import FeatureExtractor
from src.tracker import ReIDTracker
from src.ocr_reader import TrainNumberOCR

warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)


class DetectorService:
    """Сервис для управления детекторами"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector: Optional[YOLODetector] = None
        self.ppe_detector: Optional[PPEDetector] = None
        self.attribute_models: Optional[AttributeModels] = None
        self.tracker: Optional[ReIDTracker] = None
        self.ocr_reader: Optional[TrainNumberOCR] = None
        self._initialized = False
    
    def initialize(self):
        """Инициализация всех детекторов"""
        if self._initialized:
            return
        
        # Инициализация основного детектора
        yolo_cfg = self.config.get("yolo", {})
        model_path = yolo_cfg.get("model", "yolo11m.pt")
        # Если путь относительный, ищем в api_service
        if not os.path.isabs(model_path):
            # Проверяем в api_service (относительно корня api_service)
            api_service_root = Path(__file__).parent.parent.parent
            api_service_model_path = api_service_root / model_path
            if api_service_model_path.exists():
                model_path = str(api_service_model_path)
            # Иначе используем как есть (YOLO сам найдет)
        
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=self.config.get("detection", {}).get("confidence_threshold", 0.5),
            device=yolo_cfg.get("device", "cpu"),
            custom_colors=self.config.get("colors", {}),
            half_precision=self.config.get("processing", {}).get("half_precision", False)
        )
        
        # Инициализация PPE детектора
        ppe_cfg = self.config.get("ppe_detection", {})
        if ppe_cfg.get("enabled", False):
            ppe_model_path = ppe_cfg.get("model_path")
            if ppe_model_path and os.path.exists(ppe_model_path):
                try:
                    self.ppe_detector = PPEDetector(
                        model_path=ppe_model_path,
                        conf_threshold=ppe_cfg.get("confidence_threshold", 0.5),
                        device=yolo_cfg.get("device", "cpu"),
                        custom_colors=ppe_cfg.get("colors", {}),
                        half_precision=self.config.get("processing", {}).get("half_precision", False)
                    )
                except Exception as e:
                    print(f"Ошибка при инициализации PPE детектора: {e}")
        
        # Инициализация моделей атрибутов
        attr_cfg = self.config.get("attributes", {})
        if attr_cfg.get("enabled", False):
            try:
                ppe_model_path = attr_cfg.get("ppe_model")
                clothes_model_path = attr_cfg.get("clothes_model")
                
                # Проверяем пути к моделям (относительно корня api_service)
                api_service_root = Path(__file__).parent.parent.parent
                if ppe_model_path and not os.path.isabs(ppe_model_path):
                    api_service_model_path = api_service_root / ppe_model_path
                    if api_service_model_path.exists():
                        ppe_model_path = str(api_service_model_path)
                
                if clothes_model_path and not os.path.isabs(clothes_model_path):
                    api_service_model_path = api_service_root / clothes_model_path
                    if api_service_model_path.exists():
                        clothes_model_path = str(api_service_model_path)
                
                attr_device = attr_cfg.get("device", yolo_cfg.get("device", "cpu"))
                attr_conf = attr_cfg.get("confidence", 0.25)
                
                self.attribute_models = AttributeModels(
                    ppe_model_path=ppe_model_path,
                    clothes_model_path=clothes_model_path,
                    device=attr_device,
                    conf=attr_conf
                )
            except Exception as e:
                print(f"Ошибка при инициализации моделей атрибутов: {e}")
        
        # Инициализация трекера
        reid_cfg = self.config.get("re_identification", {})
        if reid_cfg.get("enabled", False):
            try:
                device = yolo_cfg.get("device", "cpu")
                feature_extractor = FeatureExtractor(device=device)
                self.tracker = ReIDTracker(
                    feature_extractor=feature_extractor,
                    max_distance=reid_cfg.get("max_distance", 0.5),
                    max_age=reid_cfg.get("max_age", 150),
                    min_hits=reid_cfg.get("min_hits", 1),
                    iou_threshold=reid_cfg.get("iou_threshold", 0.3)
                )
            except Exception as e:
                print(f"Ошибка при инициализации трекера: {e}")
        
        # Инициализация OCR (может быть долгой, поэтому делаем в конце)
        ocr_cfg = self.config.get("train_number_ocr", {})
        if ocr_cfg.get("enabled", False):
            try:
                from src.ocr_reader import TrainNumberOCR
                print("Инициализация OCR (это может занять некоторое время)...")
                self.ocr_reader = TrainNumberOCR(
                    ocr_engine=ocr_cfg.get("engine", "easyocr"),
                    languages=ocr_cfg.get("languages", ["en", "ru"]),
                    allowed_chars=ocr_cfg.get("allowed_chars", "0123456789ЭП"),
                    expected_length=ocr_cfg.get("expected_length", 7)
                )
                if self.ocr_reader.reader is not None:
                    print("OCR инициализирован успешно")
                else:
                    print("OCR не инициализирован (ошибка загрузки)")
            except Exception as e:
                print(f"Ошибка при инициализации OCR: {e}")
                import traceback
                traceback.print_exc()
                self.ocr_reader = None
        
        self._initialized = True
    
    def get_models_status(self) -> Dict[str, bool]:
        """Получить статус загрузки моделей"""
        return {
            "detector": self.detector is not None,
            "ppe_detector": self.ppe_detector is not None and self.ppe_detector.is_enabled(),
            "attribute_models": self.attribute_models is not None and self.attribute_models.is_enabled(),
            "tracker": self.tracker is not None,
            "ocr_reader": self.ocr_reader is not None
        }


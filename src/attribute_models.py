"""
Модуль для детекции атрибутов (PPE и одежда) на рабочих
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional


# Маппинг ID классов PPE на имена
PPE_NAMES = {
    0: 'boots',
    1: 'gloves',
    2: 'helmet',
    3: 'noBoots',
    4: 'noGloves',
    5: 'noHelmet',
    6: 'noVest',
    7: 'vest'
}

# Маппинг ID классов одежды на имена
CLOTHES_NAMES = {
    3: 'hat',
    4: 'jacket',
    5: 'pants',
    6: 'shirt',
    7: 'shoe',
    8: 'shorts'
}


def map_ppe_names(pairs: List[Tuple[int, float]]) -> List[str]:
    """
    Преобразует пары (class_id, confidence) в имена PPE
    
    Args:
        pairs: список кортежей (class_id, confidence)
    
    Returns:
        список имен PPE
    """
    return [PPE_NAMES.get(int(cid), f"ppe_{cid}") for cid, _ in pairs]


def map_clothes_names(pairs: List[Tuple[int, float]]) -> List[str]:
    """
    Преобразует пары (class_id, confidence) в имена одежды
    
    Args:
        pairs: список кортежей (class_id, confidence)
    
    Returns:
        список имен одежды
    """
    return [CLOTHES_NAMES.get(int(cid), f"cloth_{cid}") for cid, _ in pairs]


class AttributeModels:
    """
    Класс для работы с моделями детекции атрибутов (PPE и одежда)
    """
    
    def __init__(self, ppe_model_path: Optional[str] = None, 
                 clothes_model_path: Optional[str] = None,
                 device: str = "cpu", 
                 conf: float = 0.25):
        """
        Инициализация моделей атрибутов
        
        Args:
            ppe_model_path: путь к модели PPE (ppe_best.pt)
            clothes_model_path: путь к модели одежды (clothes_best.pt)
            device: устройство для обработки (cpu, cuda)
            conf: порог уверенности
        """
        self.device = device
        self.conf = conf
        
        # Загрузка модели PPE
        self.ppe = None
        if ppe_model_path:
            try:
                print(f"Загрузка модели PPE: {ppe_model_path}")
                self.ppe = YOLO(ppe_model_path)
                self.ppe.to(device)
                print(f"Модель PPE загружена успешно!")
            except Exception as e:
                print(f"Ошибка при загрузке модели PPE: {e}")
                self.ppe = None
        
        # Загрузка модели одежды
        self.clothes = None
        if clothes_model_path:
            try:
                print(f"Загрузка модели одежды: {clothes_model_path}")
                self.clothes = YOLO(clothes_model_path)
                self.clothes.to(device)
                print(f"Модель одежды загружена успешно!")
            except Exception as e:
                print(f"Ошибка при загрузке модели одежды: {e}")
                self.clothes = None
    
    def _infer(self, model, crop: np.ndarray) -> List[Tuple[int, float]]:
        """
        Выполняет инференс модели на обрезанном изображении
        
        Args:
            model: модель YOLO
            crop: обрезанное изображение (ROI)
        
        Returns:
            список кортежей (class_id, confidence)
        """
        if model is None:
            return []
        
        try:
            results = model(crop, device=self.device, conf=self.conf, verbose=False)
            if len(results) == 0:
                return []
            
            res = results[0]
            out = []
            for box in res.boxes:
                cls = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], 'cpu') else int(box.cls[0])
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
                out.append((cls, conf))
            return out
        except Exception as e:
            return []
    
    def run_ppe(self, crop: np.ndarray) -> List[Tuple[int, float]]:
        """
        Детекция PPE на обрезанном изображении человека
        
        Args:
            crop: обрезанное изображение человека (ROI)
        
        Returns:
            список кортежей (class_id, confidence)
        """
        return self._infer(self.ppe, crop)
    
    def run_clothes(self, crop: np.ndarray) -> List[Tuple[int, float]]:
        """
        Детекция одежды на обрезанном изображении человека
        
        Args:
            crop: обрезанное изображение человека (ROI)
        
        Returns:
            список кортежей (class_id, confidence)
        """
        return self._infer(self.clothes, crop)
    
    def is_enabled(self) -> bool:
        """Проверяет, включены ли модели"""
        return self.ppe is not None or self.clothes is not None


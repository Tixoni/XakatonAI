import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from src.image_utils import extract_roi


class FeatureExtractor:
    def __init__(self, device="cpu", feature_dim=128):

        self.device = device
        self.feature_dim = feature_dim
        self.model = self._build_model()
        self.model.eval()
        self.model.to(device)
        
        # Трансформации для предобработки изображений
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Стандартный размер для person re-id
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self):
        # Используем упрощенную архитектуру на основе ResNet
        # Можно заменить на предобученную модель для person re-id
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim)
        )
        return model
    
    def extract_features(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:

        if len(detections) == 0:
            return np.array([])
        
        features_list = []
        
        for det in detections:
            class_id, confidence, x1, y1, x2, y2 = det
            
            # Извлекаем ROI
            roi = extract_roi(frame, x1, y1, x2, y2)
            if roi is None or roi.size == 0:
                # Если ROI пустой, используем нулевой вектор
                features_list.append(np.zeros(self.feature_dim))
                continue
            
            # Конвертируем BGR в RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Предобработка и извлечение признаков
            try:
                # Преобразуем в тензор
                input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
                
                # Извлекаем признаки
                with torch.no_grad():
                    features = self.model(input_tensor)
                    # Нормализуем признаки
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                    features = features.cpu().numpy().flatten()
                
                features_list.append(features)
            except Exception as e:
                # В случае ошибки используем нулевой вектор
                print(f"Ошибка при извлечении признаков: {e}")
                features_list.append(np.zeros(self.feature_dim))
        
        return np.array(features_list)
    
    def compute_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:

        if features1.shape != features2.shape:
            return 1.0
        
        # Cosine distance
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm1 * norm2)
        # Преобразуем similarity в distance
        distance = 1.0 - cosine_similarity
        
        return max(0.0, min(1.0, distance))
    
    def compute_distance_matrix(self, features1: np.ndarray, features2: np.ndarray) -> np.ndarray:

        if len(features1) == 0 or len(features2) == 0:
            return np.array([])
        
        # Нормализуем признаки
        features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
        features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
        
        # Вычисляем cosine distance через dot product
        similarity_matrix = np.dot(features1_norm, features2_norm.T)
        distance_matrix = 1.0 - similarity_matrix
        
        return np.clip(distance_matrix, 0.0, 1.0)


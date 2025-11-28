"""
Модуль для распознавания номеров поездов с помощью OCR
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import re


class TrainNumberOCR:
    """Класс для распознавания номеров поездов"""
    
    def __init__(self, ocr_engine="easyocr", languages=['en', 'ru']):
        """
        Инициализация OCR
        
        Args:
            ocr_engine: движок OCR ('easyocr' или 'tesseract')
            languages: языки для распознавания
        """
        self.ocr_engine = ocr_engine
        self.languages = languages
        self.reader = None
        
        if ocr_engine == "easyocr":
            try:
                import easyocr
                self.reader = easyocr.Reader(languages, gpu=False)
                print(f"EasyOCR инициализирован для языков: {languages}")
            except ImportError:
                print("EasyOCR не установлен. Установите: pip install easyocr")
                self.reader = None
            except Exception as e:
                print(f"Ошибка при инициализации EasyOCR: {e}")
                self.reader = None
        elif ocr_engine == "tesseract":
            try:
                import pytesseract
                self.reader = pytesseract
                print("Tesseract OCR готов к использованию")
            except ImportError:
                print("pytesseract не установлен. Установите: pip install pytesseract")
                self.reader = None
            except Exception as e:
                print(f"Ошибка при инициализации Tesseract: {e}")
                self.reader = None
    
    def preprocess_image(self, roi: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения распознавания
        
        Args:
            roi: область интереса
            
        Returns:
            обработанное изображение
        """
        # Конвертируем в grayscale если нужно
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Увеличиваем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Бинаризация (адаптивная)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Морфологические операции для очистки
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Увеличение размера для лучшего распознавания
        height, width = cleaned.shape
        scale = max(2.0, 300.0 / max(height, width))
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(cleaned, (new_width, new_height), 
                            interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def recognize_text_easyocr(self, roi: np.ndarray) -> Optional[str]:
        """Распознавание текста с помощью EasyOCR"""
        if self.reader is None:
            return None
        
        try:
            # Предобработка
            processed = self.preprocess_image(roi)
            
            # Распознавание
            results = self.reader.readtext(processed)
            
            if not results:
                return None
            
            # Объединяем все найденные тексты
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Минимальная уверенность
                    texts.append(text.strip())
            
            if texts:
                combined_text = " ".join(texts)
                # Очищаем текст от лишних символов
                cleaned = re.sub(r'[^\w\s]', '', combined_text)
                return cleaned.strip()
            
            return None
        except Exception as e:
            print(f"Ошибка EasyOCR: {e}")
            return None
    
    def recognize_text_tesseract(self, roi: np.ndarray) -> Optional[str]:
        """Распознавание текста с помощью Tesseract"""
        if self.reader is None:
            return None
        
        try:
            # Предобработка
            processed = self.preprocess_image(roi)
            
            # Конфигурация для распознавания цифр и букв
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
            
            # Распознавание
            text = self.reader.image_to_string(processed, config=config, lang='eng+rus')
            
            if text:
                # Очищаем текст
                cleaned = re.sub(r'[^\w\s]', '', text)
                return cleaned.strip()
            
            return None
        except Exception as e:
            print(f"Ошибка Tesseract: {e}")
            return None
    
    def recognize_train_number(self, frame: np.ndarray, 
                              roi_config: Dict) -> Optional[str]:
        """
        Распознавание номера поезда из заданной области кадра
        
        Args:
            frame: кадр изображения
            roi_config: конфигурация области интереса
                {
                    "x": процент от ширины (0.0-1.0),
                    "y": процент от высоты (0.0-1.0),
                    "width": процент ширины (0.0-1.0),
                    "height": процент высоты (0.0-1.0)
                }
            
        Returns:
            распознанный номер поезда или None
        """
        if self.reader is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Вычисляем координаты ROI
        x = int(w * roi_config.get("x", 0.0))
        y = int(h * roi_config.get("y", 0.0))
        width = int(w * roi_config.get("width", 0.2))
        height = int(h * roi_config.get("height", 0.1))
        
        # Проверяем границы
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        width = max(1, min(w - x, width))
        height = max(1, min(h - y, height))
        
        # Извлекаем ROI
        roi = frame[y:y+height, x:x+width]
        
        if roi.size == 0:
            return None
        
        # Распознавание в зависимости от движка
        if self.ocr_engine == "easyocr":
            return self.recognize_text_easyocr(roi)
        elif self.ocr_engine == "tesseract":
            return self.recognize_text_tesseract(roi)
        
        return None
    
    def recognize_from_train_bbox(self, frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int],
                                 roi_offset: Dict = None) -> Optional[str]:
        """
        Распознавание номера поезда из области детектированного поезда
        
        Args:
            frame: кадр изображения
            bbox: координаты поезда (x1, y1, x2, y2)
            roi_offset: смещение и размер ROI относительно bbox
                {
                    "x_offset": смещение по X (процент от ширины bbox),
                    "y_offset": смещение по Y (процент от высоты bbox),
                    "width": ширина ROI (процент от ширины bbox),
                    "height": высота ROI (процент от высоты bbox)
                }
        
        Returns:
            распознанный номер поезда или None
        """
        if self.reader is None:
            return None
        
        x1, y1, x2, y2 = bbox
        train_width = x2 - x1
        train_height = y2 - y1
        
        # Параметры по умолчанию для номера в верхней части поезда
        if roi_offset is None:
            roi_offset = {
                "x_offset": 0.1,  # 10% от левого края
                "y_offset": 0.05,  # 5% от верхнего края
                "width": 0.3,      # 30% ширины поезда
                "height": 0.15     # 15% высоты поезда
            }
        
        # Вычисляем координаты ROI
        roi_x = int(x1 + train_width * roi_offset["x_offset"])
        roi_y = int(y1 + train_height * roi_offset["y_offset"])
        roi_w = int(train_width * roi_offset["width"])
        roi_h = int(train_height * roi_offset["height"])
        
        # Проверяем границы
        h, w = frame.shape[:2]
        roi_x = max(0, min(w - 1, roi_x))
        roi_y = max(0, min(h - 1, roi_y))
        roi_w = max(1, min(w - roi_x, roi_w))
        roi_h = max(1, min(h - roi_y, roi_h))
        
        # Извлекаем ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if roi.size == 0:
            return None
        
        # Распознавание
        if self.ocr_engine == "easyocr":
            return self.recognize_text_easyocr(roi)
        elif self.ocr_engine == "tesseract":
            return self.recognize_text_tesseract(roi)
        
        return None


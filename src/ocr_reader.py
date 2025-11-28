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
                import warnings
                # Подавляем предупреждение о pin_memory при использовании CPU
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
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
            # Пробуем распознавание без предобработки (для лучшего качества)
            results_raw = self.reader.readtext(roi)
            
            # Если не получилось, пробуем с предобработкой
            if not results_raw or len(results_raw) == 0:
                processed = self.preprocess_image(roi)
                results = self.reader.readtext(processed)
            else:
                results = results_raw
            
            if not results:
                return None
            
            # Объединяем все найденные тексты, сортируем по уверенности
            texts_with_conf = []
            for (bbox, text, confidence) in results:
                if confidence > 0.2:  # Снижаем порог для лучшего распознавания
                    texts_with_conf.append((text.strip(), confidence))
            
            if texts_with_conf:
                # Сортируем по уверенности (от большей к меньшей)
                texts_with_conf.sort(key=lambda x: x[1], reverse=True)
                
                # Берем текст с наибольшей уверенностью или объединяем несколько
                if len(texts_with_conf) == 1:
                    combined_text = texts_with_conf[0][0]
                else:
                    # Объединяем несколько результатов
                    combined_text = " ".join([t[0] for t in texts_with_conf[:3]])  # Берем до 3 лучших
                
                # Очищаем текст, но сохраняем пробелы и кириллицу
                # Убираем только специальные символы, оставляем буквы, цифры и пробелы
                cleaned = re.sub(r'[^\w\sА-Яа-яЁё]', '', combined_text)
                cleaned = re.sub(r'\s+', ' ', cleaned)  # Убираем множественные пробелы
                result = cleaned.strip()
                
                # Проверяем, что результат не пустой и содержит хотя бы одну букву или цифру
                if result and (re.search(r'[А-Яа-яЁёA-Za-z]', result) or re.search(r'\d', result)):
                    return result
            
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
    
    def recognize_from_right_half(self, frame: np.ndarray) -> Optional[str]:
        """
        Распознавание номера поезда из правой половины экрана
        (разделение вертикальной чертой по середине, ищем в правой половине)
        
        Args:
            frame: кадр изображения
            
        Returns:
            распознанный номер поезда или None
        """
        if self.reader is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Центр экрана - вертикальная черта по середине
        center_x = w // 2
        
        # Правая половина экрана: от центра до края по всей высоте
        x = center_x  # Начинаем с центра по X
        y = 0  # Начинаем с верха экрана
        width = w - center_x  # Вся правая половина ширины
        height = h  # Вся высота экрана
        
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
    
    def recognize_from_bottom_right_quadrant(self, frame: np.ndarray) -> Optional[str]:
        """
        Распознавание номера поезда из правого нижнего квадранта кадра
        (устаревший метод, используйте recognize_from_right_half)
        """
        return self.recognize_from_right_half(frame)
    
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


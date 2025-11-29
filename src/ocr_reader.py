"""
Модуль для распознавания номеров поездов с помощью OCR
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import re

# Замена похожих символов, которые OCR часто путает
CHAR_REPLACEMENTS = {
    # Латинские буквы, похожие на цифры
    'O': '0',  # Латинская O -> 0
    'o': '0',  # Строчная o -> 0
    'I': '1',  # I -> 1
    'l': '1',  # l -> 1
    '|': '1',  # | -> 1
    'S': '5',  # S -> 5
    's': '5',
    'D': '0',  # D -> 0 (иногда)
    'd': '0',
    'G': '6',  # G -> 6 (иногда)
    'g': '6',
    'Q': '0',  # Q -> 0
    'q': '0',
    # Кириллические буквы, похожие на цифры
    'О': '0',  # Кириллическая О -> 0
    'о': '0',  # Строчная о -> 0
    'З': '3',  # З -> 3
    'з': '3',
    'Б': '6',  # Б -> 6
    'б': '6',
    'В': '8',  # В -> 8 (иногда)
    'в': '8',
}



class TrainNumberOCR:
    """Класс для распознавания номеров поездов"""
    
    def __init__(self, ocr_engine="easyocr", languages=['en', 'ru'], 
                 allowed_chars="0123456789ЭП", expected_length=7):
        """
        Инициализация OCR
        
        Args:
            ocr_engine: движок OCR ('easyocr' или 'tesseract')
            languages: языки для распознавания
            allowed_chars: строка допустимых символов для номера поезда
            expected_length: ожидаемая длина номера поезда
        """
        self.ocr_engine = ocr_engine
        self.languages = languages
        self.allowed_chars = set(allowed_chars)
        self.expected_length = expected_length
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
    
    def filter_train_number(self, text: str) -> Optional[str]:
        """
        Фильтрует текст, оставляя только допустимые символы для номера поезда.
        Также заменяет похожие символы, которые OCR часто путает.
        
        Args:
            text: исходный текст
            
        Returns:
            отфильтрованный текст или None, если не осталось допустимых символов
        """
        if not text:
            return None
        
        # Сначала заменяем похожие символы
        replaced = ''
        for c in text:
            if c in CHAR_REPLACEMENTS:
                replaced += CHAR_REPLACEMENTS[c]
            else:
                replaced += c
        
        # Затем оставляем только допустимые символы (из конфига)
        filtered = ''.join(c for c in replaced if c in self.allowed_chars)
        
        # Убираем пробелы и проверяем, что осталось что-то
        filtered = filtered.strip()
        
        if filtered and len(filtered) > 0:
            # Если длина соответствует ожидаемой, возвращаем как есть
            if len(filtered) == self.expected_length:
                return filtered
            # Если длина больше ожидаемой, берем первые expected_length символов
            elif len(filtered) > self.expected_length:
                return filtered[:self.expected_length]
            # Если длина меньше, но есть символы, возвращаем (может быть неполный номер)
            else:
                return filtered
        
        return None
    
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
            
            # Собираем все распознанные тексты с их позициями
            texts_with_pos = []
            for (bbox, text, confidence) in results:
                if confidence > 0.2:  # Снижаем порог для лучшего распознавания
                    # Вычисляем минимальную x координату (левый край) для сортировки слева направо
                    # bbox - это numpy array формы (4, 2) с точками [x, y]
                    if isinstance(bbox, np.ndarray):
                        x_min = float(np.min(bbox[:, 0]))  # Минимальная x координата
                    else:
                        # Если это список, берем минимальную x из всех точек
                        x_min = min(point[0] for point in bbox)
                    texts_with_pos.append((x_min, text.strip(), confidence))
            
            if texts_with_pos:
                # Сортируем по позиции (x координате) слева направо для правильного порядка
                texts_with_pos.sort(key=lambda x: x[0])
                
                # Объединяем тексты в правильном порядке
                combined_text = ''.join([t[1] for t in texts_with_pos])
                
                # Очищаем текст, но сохраняем пробелы и кириллицу
                # Убираем только специальные символы, оставляем буквы, цифры и пробелы
                cleaned = re.sub(r'[^\w\sА-Яа-яЁё]', '', combined_text)
                cleaned = re.sub(r'\s+', '', cleaned)  # Убираем все пробелы
                result = cleaned.strip()
                
                # Фильтруем только допустимые символы для номера поезда
                filtered_result = self.filter_train_number(result)
                
                if filtered_result:
                    return filtered_result
            
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
            
            # Конфигурация для распознавания цифр и букв (используем допустимые символы из конфига)
            allowed_chars_str = ''.join(sorted(self.allowed_chars))
            config = f'--oem 3 --psm 7 -c tessedit_char_whitelist={allowed_chars_str}'
            
            # Распознавание
            text = self.reader.image_to_string(processed, config=config, lang='eng+rus')
            
            if text:
                # Фильтруем только допустимые символы для номера поезда
                filtered_result = self.filter_train_number(text)
                
                if filtered_result:
                    return filtered_result
            
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


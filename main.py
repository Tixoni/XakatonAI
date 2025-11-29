import sys
import os
import warnings
from pathlib import Path

# Подавляем предупреждение о pin_memory при использовании CPU
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

# Импортируем наши модули
from src.config_loader import load_config
from src.detector import YOLODetector
from src.processors import process_image, process_video

def main():
    print("=" * 60)
    print("Детекция людей и поездов (YOLO11) - Modular Version")
    print("=" * 60)
    
    # 1. Загрузка конфига
    config = load_config("config.json")
    
    # 2. Проверка входного файла
    input_file = config.get("input_file", "")
    if not input_file:
        print("Ошибка: не указан input_file в конфиге")
        sys.exit(1)
        
    input_file = os.path.normpath(input_file.strip('"\''))
    if not os.path.exists(input_file):
        print(f"Файл не найден: {input_file}")
        sys.exit(1)

    # 3. Инициализация детектора
    yolo_cfg = config.get("yolo", {})
    detector = YOLODetector(
        model_path=yolo_cfg.get("model", "yolo11m.pt"),
        conf_threshold=config.get("detection", {}).get("confidence_threshold", 0.5),
        device=yolo_cfg.get("device", "cpu"),
        custom_colors=config.get("colors", {}),
        half_precision=config.get("processing", {}).get("half_precision", False)
    )

    # 4. Запуск обработки
    ext = Path(input_file).suffix.lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    if ext in image_exts:
        process_image(input_file, detector, config)
    elif ext in video_exts:
        process_video(input_file, detector, config)
    else:
        print(f"Неизвестный формат файла: {ext}")

if __name__ == "__main__":
    main()
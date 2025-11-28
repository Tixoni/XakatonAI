import cv2
import os
from pathlib import Path
from src.image_utils import imread_unicode, resize_frame
from src.filters import apply_color_filters, annotate_rejected

def process_image(image_path, detector, config):
    print(f"Загрузка изображения: {image_path}")
    if not os.path.exists(image_path):
        print(f"Ошибка: файл не существует!")
        return

    image = imread_unicode(image_path)
    if image is None:
        print("Ошибка загрузки изображения")
        return

    # Настройки
    processing_cfg = config.get("processing", {})
    debug_cfg = config.get("debug", {})
    save_results = processing_cfg.get("save_results", False)
    show_preview = processing_cfg.get("show_preview", True)
    output_dir = processing_cfg.get("output_dir", "results")

    # Ресайз
    opt_config = config.get("video_optimization", {})
    max_width = opt_config.get("max_width", 1920)
    max_height = opt_config.get("max_height", 1080)
    
    # ... Логика ресайза вызывая resize_frame ...
    image, scale = resize_frame(image, max_width, max_height)

    # Детекция
    target_classes = config.get("detection", {}).get("target_classes", [0, 6])
    detections = detector.detect(image, target_classes=target_classes)
    
    # Фильтры
    color_filters = config.get("color_filters", {})
    detections, rejected = apply_color_filters(image, detections, color_filters, detector.CLASSES, debug_cfg)

    # Отрисовка
    result_image = detector.draw_detections(image, detections)
    if debug_cfg.get("show_filtered_objects") and rejected:
        result_image = annotate_rejected(result_image, rejected, detector)

    # Сохранение и показ
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir) / f"{Path(image_path).stem}_detected.png"
        cv2.imwrite(str(out_path), result_image)
        print(f"Сохранено в {out_path}")

    if show_preview:
        cv2.imshow('Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(video_path, detector, config):
    # Вставьте сюда весь код функции process_video.
    # Замените вызовы локальных функций на импортированные:
    # resize_frame(...) -> resize_frame(...) (из image_utils)
    # apply_color_filters(...) -> apply_color_filters(...) (из filters)
    pass
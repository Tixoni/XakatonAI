"""
Модуль для обработки изображений и видео
"""

import cv2
import os
from pathlib import Path
from src.image_utils import imread_unicode, resize_frame
from src.filters import apply_color_filters, annotate_rejected
from src.reid import FeatureExtractor
from src.tracker import ReIDTracker
from src.ocr_reader import TrainNumberOCR


def process_image(image_path, detector, config):
    """Обработка изображения"""
    print(f"Загрузка изображения: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Ошибка: файл не существует!")
        print(f"Проверьте путь: {image_path}")
        return
    
    image = imread_unicode(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение!")
        print(f"Убедитесь, что файл является корректным изображением")
        return
    
    print(f"Изображение загружено: {image.shape[1]}x{image.shape[0]} пикселей")
    
    processing_cfg = config.get("processing", {})
    show_preview = processing_cfg.get("show_preview", True)
    save_results = processing_cfg.get("save_results", False)
    output_dir = processing_cfg.get("output_dir", "results")
    debug_cfg = config.get("debug", {})
    
    # Оптимизация размера если нужно
    opt_config = config.get("video_optimization", {})
    max_width = opt_config.get("max_width", 1920)
    max_height = opt_config.get("max_height", 1080)
    maintain_aspect = opt_config.get("maintain_aspect_ratio", True)
    keep_width_native = opt_config.get("keep_width_native", False)
    
    needs_resize = False
    if keep_width_native and max_height and image.shape[0] > max_height:
        needs_resize = True
    elif (image.shape[1] > max_width or image.shape[0] > max_height):
        needs_resize = True
    
    if needs_resize:
        target_desc = f"до высоты {max_height}px" if keep_width_native else f"до {max_width}x{max_height}"
        print(f"Оптимизация размера {target_desc} ...")
        image, scale = resize_frame(image, max_width, max_height, maintain_aspect, keep_width_native)
        print(f"Размер изменен: {image.shape[1]}x{image.shape[0]} пикселей (масштаб: {scale:.2f})")
    
    print("Выполняется детекция...")
    
    target_classes = config.get("detection", {}).get("target_classes", [0, 6])
    detections = detector.detect(image, target_classes=target_classes)
    
    color_filters = config.get("color_filters", {})
    detections, rejected = apply_color_filters(image, detections, color_filters, detector.CLASSES, debug_cfg)
    
    print(f"Найдено объектов: {len(detections)}")
    if rejected:
        print(f"Отклонено цветовым фильтром: {len(rejected)}")
    for class_id, confidence, x1, y1, x2, y2 in detections:
        class_name = detector.CLASSES.get(class_id, "unknown")
        print(f"  - {class_name}: {confidence:.2f}")
    
    result_image = detector.draw_detections(image, detections)
    
    if debug_cfg.get("show_filtered_objects") and rejected:
        result_image = annotate_rejected(result_image, rejected, detector)
    
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{Path(image_path).stem}_detected.png"
        cv2.imwrite(str(output_path), result_image)
        print(f"Результат сохранен: {output_path}")
    
    if show_preview:
        cv2.imshow('Детекция объектов', result_image)
        print("\nНажмите любую клавишу для закрытия окна...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path, detector, config):
    """Обработка видео с оптимизацией"""
    # Для путей с кириллицей
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        try:
            video_path_alt = str(Path(video_path).absolute())
            cap = cv2.VideoCapture(video_path_alt)
        except:
            pass
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        print("Убедитесь, что путь указан правильно и файл существует")
        return
    
    # Получаем параметры видео
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Оригинальное видео: {width}x{height}, FPS: {original_fps}, Кадров: {total_frames}")
    
    # Параметры оптимизации
    opt_config = config.get("video_optimization", {})
    target_fps = opt_config.get("target_fps", 10)
    max_width = opt_config.get("max_width", 1280)
    max_height = opt_config.get("max_height", 720)
    frame_skip = opt_config.get("frame_skip", 1)
    maintain_aspect = opt_config.get("maintain_aspect_ratio", True)
    keep_width_native = opt_config.get("keep_width_native", False)
    
    processing_cfg = config.get("processing", {})
    show_preview = processing_cfg.get("show_preview", True)
    save_results = processing_cfg.get("save_results", False)
    output_dir = processing_cfg.get("output_dir", "results")
    debug_cfg = config.get("debug", {})
    
    # Вычисляем, через сколько кадров обрабатывать
    # Если frame_skip явно указан > 1, используем его, иначе вычисляем на основе target_fps
    if frame_skip > 1:
        skip_frames = frame_skip
        print(f"Используется явный пропуск кадров: {skip_frames}")
    else:
        # Вычисляем пропуск на основе целевого FPS
        if target_fps <= 0 or original_fps <= 0:
            skip_frames = 1
        elif target_fps < original_fps:
            skip_frames = max(1, int(original_fps / target_fps))
        else:
            skip_frames = 1
        print(f"Автоматический пропуск кадров для FPS {target_fps}: {skip_frames}")
    
    # Предварительная оценка размера вывода
    ret, temp_frame = cap.read()
    if not ret or temp_frame is None:
        print("Не удалось прочитать первый кадр.")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    resized_sample, _ = resize_frame(temp_frame, max_width, max_height, maintain_aspect, keep_width_native)
    display_width = resized_sample.shape[1]
    display_height = resized_sample.shape[0]
    
    print(f"Обработка: пропуск кадров {skip_frames}, размер {display_width}x{display_height}")
    print(f"Целевой FPS: {target_fps}")
    print("\nНажмите 'q' для выхода\n")
    
    target_classes = config.get("detection", {}).get("target_classes", [0, 6])
    detections_count = {cid: 0 for cid in target_classes}
    color_filters = config.get("color_filters", {})
    rejected_total = 0
    writer = None
    output_path = None
    
    # Инициализация re-identification трекера
    reid_cfg = config.get("re_identification", {})
    use_reid = reid_cfg.get("enabled", False)
    tracker = None
    feature_extractor = None
    
    if use_reid:
        print("Инициализация re-identification трекера...")
        try:
            device = config.get("yolo", {}).get("device", "cpu")
            feature_extractor = FeatureExtractor(device=device)
            tracker = ReIDTracker(
                feature_extractor=feature_extractor,
                max_distance=reid_cfg.get("max_distance", 0.5),
                max_age=reid_cfg.get("max_age", 30),
                min_hits=reid_cfg.get("min_hits", 1),
                iou_threshold=reid_cfg.get("iou_threshold", 0.3)
            )
            print("Re-identification трекер инициализирован успешно")
        except Exception as e:
            print(f"Ошибка при инициализации re-identification: {e}")
            print("Продолжаем без re-identification...")
            use_reid = False
            tracker = None
    
    # Инициализация OCR для распознавания номеров поездов
    ocr_cfg = config.get("train_number_ocr", {})
    use_ocr = ocr_cfg.get("enabled", False)
    ocr_reader = None
    
    if use_ocr:
        print("Инициализация OCR для распознавания номеров поездов...")
        try:
            ocr_engine = ocr_cfg.get("engine", "easyocr")
            ocr_reader = TrainNumberOCR(ocr_engine=ocr_engine)
            if ocr_reader.reader is None:
                print("OCR не инициализирован, продолжаем без распознавания номеров...")
                use_ocr = False
            else:
                print("OCR инициализирован успешно")
        except Exception as e:
            print(f"Ошибка при инициализации OCR: {e}")
            print("Продолжаем без распознавания номеров...")
            use_ocr = False
            ocr_reader = None
    
    frame_count = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Пропускаем кадры для снижения FPS
            if frame_count % skip_frames != 0:
                continue
            
            processed_count += 1
            
            resize_needed = False
            if keep_width_native:
                if max_height and frame.shape[0] > max_height:
                    resize_needed = True
            else:
                if frame.shape[1] > max_width or frame.shape[0] > max_height:
                    resize_needed = True
            
            if resize_needed:
                resized_frame, _ = resize_frame(frame, max_width, max_height, maintain_aspect, keep_width_native)
            else:
                resized_frame = frame
            
            # Детекция
            detections = detector.detect(resized_frame, target_classes=target_classes)
            detections, rejected = apply_color_filters(resized_frame, detections, color_filters, detector.CLASSES, debug_cfg)
            rejected_total += len(rejected)
            
            # Re-identification трекинг
            tracks = None
            train_numbers = {}  # Словарь {track_id: train_number}
            
            if use_reid and tracker is not None:
                try:
                    tracks = tracker.update(resized_frame, detections)
                    
                    # Распознавание номеров поездов
                    if use_ocr and ocr_reader is not None:
                        # Проверяем, используется ли фиксированная область для номера
                        fixed_roi = ocr_cfg.get("fixed_roi", None)
                        
                        if fixed_roi and fixed_roi.get("enabled", False):
                            # Используем фиксированную область кадра
                            train_number = ocr_reader.recognize_train_number(
                                resized_frame, fixed_roi
                            )
                            if train_number:
                                # Присваиваем номер всем поездам в кадре
                                for track_id, class_id, _, _, _, _, _ in tracks:
                                    if class_id == 6:  # train
                                        track = tracker.tracks.get(track_id)
                                        if track:
                                            track.train_number = train_number
                                            train_numbers[track_id] = train_number
                                if train_number and processed_count % 30 == 0:
                                    print(f"Распознан номер поезда из фиксированной области: {train_number}")
                        elif tracks:
                            # Используем область детектированного поезда
                            for track_id, class_id, confidence, x1, y1, x2, y2 in tracks:
                                # Распознаем номер только для поездов (class_id = 6)
                                if class_id == 6:  # train
                                    track = tracker.tracks.get(track_id)
                                    if track and track.train_number is None:
                                        # Пытаемся распознать номер только один раз для каждого трека
                                        try:
                                            roi_offset = ocr_cfg.get("roi_offset", None)
                                            train_number = ocr_reader.recognize_from_train_bbox(
                                                resized_frame, (x1, y1, x2, y2), roi_offset
                                            )
                                            if train_number:
                                                track.train_number = train_number
                                                print(f"Распознан номер поезда ID:{track_id}: {train_number}")
                                        except Exception as e:
                                            if debug_cfg.get("log_detection_details"):
                                                print(f"Ошибка OCR для трека {track_id}: {e}")
                                    
                                    # Добавляем номер в словарь для отрисовки
                                    if track and track.train_number:
                                        train_numbers[track_id] = track.train_number
                    
                    if tracks and len(tracks) > 0:
                        # Отрисовка с ID треков и номерами поездов
                        result_frame = detector.draw_detections(
                            resized_frame, tracks, show_track_ids=True, train_numbers=train_numbers
                        )
                    else:
                        # Если треки еще не созданы, показываем детекции без ID
                        result_frame = detector.draw_detections(resized_frame, detections, show_track_ids=False)
                except Exception as e:
                    print(f"Ошибка в трекинге: {e}")
                    # Fallback: показываем детекции без ID
                    result_frame = detector.draw_detections(resized_frame, detections, show_track_ids=False)
            else:
                # Подсчет
                for class_id, _, _, _, _, _ in detections:
                    if class_id in detections_count:
                        detections_count[class_id] += 1
                
                # Отрисовка без ID
                result_frame = detector.draw_detections(resized_frame, detections, show_track_ids=False)
            if debug_cfg.get("show_filtered_objects") and rejected:
                result_frame = annotate_rejected(result_frame, rejected, detector, color=(0, 0, 255))
            
            # Информация на кадре
            if use_reid and tracker is not None and tracks:
                # Подсчет уникальных треков
                unique_tracks_by_class = {}
                for track_id, class_id, _, _, _, _, _ in tracks:
                    if class_id not in unique_tracks_by_class:
                        unique_tracks_by_class[class_id] = set()
                    unique_tracks_by_class[class_id].add(track_id)
                
                counts_text = " | ".join(
                    f"{detector.CLASSES.get(cid, f'class_{cid}')}: {len(unique_tracks_by_class.get(cid, set()))}"
                    for cid in target_classes
                )
                info_text = f"Кадр: {processed_count} | Треков: {len(tracks)}" + (f" | {counts_text}" if counts_text else "")
            else:
                counts_text = " | ".join(
                    f"{detector.CLASSES.get(cid, f'class_{cid}')}: {detections_count[cid]}"
                    for cid in detections_count
                )
                info_text = f"Кадр: {processed_count}" + (f" | {counts_text}" if counts_text else "")
            
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Сохранение результата
            if save_results:
                if writer is None:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = Path(output_dir) / f"{Path(video_path).stem}_detected.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps,
                                             (result_frame.shape[1], result_frame.shape[0]))
                writer.write(result_frame)
            
            # Показываем результат
            if show_preview:
                cv2.imshow('Детекция объектов', result_frame)
                
                # Задержка для соответствия целевому FPS
                delay = int(1000 / target_fps) if target_fps > 0 else 1
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
            
            # Прогресс
            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Обработано кадров: {processed_count}, прогресс: {progress:.1f}%")
    
    except KeyboardInterrupt:
        print("\nОбработка прервана")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nОбработка завершена!")
        print(f"Обработано кадров: {processed_count}")
        
        if use_reid and tracker is not None:
            # Статистика по всем когда-либо созданным трекам
            for cid in target_classes:
                class_name = detector.CLASSES.get(cid, f"class_{cid}")
                # Используем all_tracks_by_class для подсчета всех уникальных треков
                unique_count = len(tracker.all_tracks_by_class.get(cid, set()))
                active_count = len([t for t in tracker.tracks.values() if t.class_id == cid])
                print(f"Уникальных треков {class_name}: {unique_count} (активных: {active_count})")
        else:
            for cid, count in detections_count.items():
                class_name = detector.CLASSES.get(cid, f"class_{cid}")
                print(f"Обнаружено {class_name}: {count}")
        
        if rejected_total:
            print(f"Отклонено цветовым фильтром: {rejected_total}")
        if save_results and output_path:
            print(f"Видео сохранено: {output_path}")

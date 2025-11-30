
import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
from PIL import Image

from app.models import (
    DetectionRequest, DetectionResponse, HealthResponse, 
    ErrorResponse, ObjectInfo, Statistics
)
from app.services.detector_service import DetectorService
from app.services.processing_service import ProcessingService

# Версия API
API_VERSION = "1.0.0"

# Инициализация FastAPI
app = FastAPI(
    title="Detection API",
    description="API для детекции людей и поездов с использованием YOLO",
    version=API_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для сервисов
detector_service: Optional[DetectorService] = None
processing_service: Optional[ProcessingService] = None
config: dict = {}


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        # Ищем config.json в корне проекта или в api_service
        possible_paths = [
            Path(__file__).parent.parent / "config.json",  # api_service/config.json
            Path(__file__).parent.parent.parent / "config.json",  # корень проекта
            Path("config.json")  # текущая директория
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        
        if config_path is None:
            # Используем конфиг по умолчанию
            print("Конфиг не найден, используем конфиг по умолчанию")
            return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке конфига: {e}, используем конфиг по умолчанию")
        return get_default_config()


def get_default_config() -> dict:
    return {
        "yolo": {
            "model": "yolo11m.pt",
            "device": "cpu"
        },
        "detection": {
            "confidence_threshold": 0.5,
            "target_classes": [0, 6],
            "class_names": {
                "0": "person",
                "6": "train"
            }
        },
        "video_optimization": {
            "target_fps": 20,
            "max_width": 1280,
            "max_height": 540,
            "frame_skip": 1,
            "maintain_aspect_ratio": True,
            "keep_width_native": True
        },
        "processing": {
            "show_preview": False,
            "half_precision": False,
            "save_results": True,
            "output_dir": "results"
        },
        "debug": {
            "show_filtered_objects": False,
            "log_detection_details": False
        },
        "re_identification": {
            "enabled": True,
            "max_distance": 0.5,
            "max_age": 150,
            "min_hits": 10,
            "iou_threshold": 0.2,
            "use_objects": True
        },
        "train_number_ocr": {
            "enabled": True,
            "engine": "easyocr",
            "frame_skip": 10,
            "use_bottom_right_quadrant": True,
            "allowed_chars": "0123456789ЭП",
            "expected_length": 7,
            "languages": ["en", "ru"]
        },
        "ppe_detection": {
            "enabled": False,
            "model_path": None,
            "confidence_threshold": 0.5,
            "target_classes": None
        },
        "attributes": {
            "enabled": True,
            "ppe_model": "ppe_best.pt",
            "clothes_model": "clothes_best.pt",
            "device": "cpu",
            "confidence": 0.25,
            "update_interval": 10
        }
    }


@app.on_event("startup")
async def startup_event():
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    global detector_service, processing_service, config
    
    print("Инициализация сервисов...")
    
    try:
        # Загрузка конфига - быстрая операция
        config = load_config()
        
        # Инициализация сервисов в отдельном потоке, чтобы не блокировать event loop
        def init_services():
            try:
                global detector_service, processing_service
                detector_service = DetectorService(config)
                detector_service.initialize()
                processing_service = ProcessingService(detector_service, config)
                return True
            except Exception as e:
                print(f"Ошибка в init_services: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Выполняем в executor с таймаутом
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="init")
        
        try:
            # Таймаут 5 минут на инициализацию (модели могут загружаться долго)
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, init_services),
                timeout=300.0
            )
            if not result:
                raise Exception("Инициализация сервисов завершилась с ошибкой")
        except asyncio.TimeoutError:
            print("Таймаут при инициализации сервисов (превышено 5 минут)")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception as e:
            print(f"Ошибка при инициализации: {e}")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False)
        
        print("Сервисы инициализированы успешно!")
    except Exception as e:
        print(f"Критическая ошибка при инициализации сервисов: {e}")
        import traceback
        traceback.print_exc()
        # Инициализируем пустые сервисы, чтобы приложение не упало
        detector_service = None
        processing_service = None
        config = get_default_config()


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Detection API",
        "version": API_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if detector_service is None:
        return HealthResponse(
            status="error",
            version=API_VERSION,
            models_loaded={}
        )
    
    try:
        models_status = detector_service.get_models_status()
        # Проверяем, что хотя бы основной детектор загружен
        is_healthy = models_status.get("detector", False)
        
        return HealthResponse(
            status="healthy" if is_healthy else "degraded",
            version=API_VERSION,
            models_loaded=models_status
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            version=API_VERSION,
            models_loaded={}
        )


def decode_image(file_content: bytes) -> np.ndarray:
    if not file_content or len(file_content) == 0:
        raise ValueError("Пустой файл")
    nparr = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось декодировать изображение. Убедитесь, что файл является корректным изображением (JPEG, PNG и т.д.)")
    if image.size == 0:
        raise ValueError("Изображение пустое после декодирования")
    return image


def encode_image(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    target_classes: str = Form("[0, 6]"),  # JSON строка
    return_image: bool = Form(True),
    return_statistics: bool = Form(True)
):
    
    if detector_service is None or processing_service is None:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    
    try:
        # Проверка типа файла
        if not file.content_type:
            # Если content_type не указан, проверяем расширение
            if file.filename:
                ext = Path(file.filename).suffix.lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                    raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла. Используйте JPEG, PNG, BMP, GIF или WebP")
        elif not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        # Чтение файла
        file_content = await file.read()
        if not file_content or len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        
        image = decode_image(file_content)
        
        # Парсинг target_classes
        try:
            import json
            target_classes_list = json.loads(target_classes)
        except:
            target_classes_list = [0, 6]
        
        # Параметры запроса
        request_params = {
            "confidence_threshold": confidence_threshold,
            "target_classes": target_classes_list,
            "return_image": return_image,
            "return_statistics": return_statistics
        }
        
        # Обработка
        objects, result_image, metadata = processing_service.process_image(
            image, request_params
        )
        
        # Формирование ответа
        response_data = {
            "success": True,
            "message": f"Найдено объектов: {len(objects)}",
            "objects": objects,
            "processing_time": metadata.get("processing_time"),
        }
        
        # Добавление изображения
        if return_image and result_image is not None:
            response_data["image"] = encode_image(result_image)
        
        # Добавление статистики
        if return_statistics:
            stats = metadata.get("statistics", {})
            if stats:
                response_data["statistics"] = {
                    obj_type: Statistics(**stat_data) 
                    for obj_type, stat_data in stats.items()
                }
            else:
                response_data["statistics"] = None
        
        return DetectionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    target_classes: str = Form("[0, 6]"),
    return_video: bool = Form(True),
    return_statistics: bool = Form(True)
):

    if detector_service is None or processing_service is None:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован. Проверьте /health")
    
    try:
        # Логируем информацию о файле для отладки
        print(f"Received video file: filename={file.filename}, content_type={file.content_type}")
        
        # Проверка типа файла - более мягкая проверка
        is_video = False
        
        # Проверяем расширение файла
        if file.filename:
            ext = Path(file.filename).suffix.lower()
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp']
            if ext in video_extensions:
                is_video = True
                print(f"Video extension detected: {ext}")
        
        # Проверяем content_type
        if file.content_type:
            if file.content_type.startswith('video/'):
                is_video = True
                print(f"Video content_type detected: {file.content_type}")
            elif file.content_type.startswith('application/'):
                # Некоторые клиенты отправляют video как application/octet-stream
                if is_video:  # Если расширение правильное, принимаем
                    print(f"Accepting application/* with video extension")
                    is_video = True
        
        # Если не определили как видео, выдаем ошибку
        if not is_video:
            error_msg = f"File must be a video. filename={file.filename}, content_type={file.content_type}"
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Сохранение временного файла
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        filename = file.filename or "upload_video"
        temp_file = temp_dir / f"upload_{int(time.time())}_{filename}"
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            if not content or len(content) == 0:
                raise HTTPException(status_code=400, detail="Файл пустой")
            f.write(content)
        
        # Парсинг target_classes
        try:
            import json
            target_classes_list = json.loads(target_classes)
        except:
            target_classes_list = [0, 6]
        
        # Параметры запроса
        request_params = {
            "confidence_threshold": confidence_threshold,
            "target_classes": target_classes_list,
            "return_video": return_video,
            "return_statistics": return_statistics
        }
        
        # Обработка
        objects, output_path, metadata = processing_service.process_video(
            str(temp_file), request_params
        )
        
        # Если нужно вернуть только статистику
        if not return_video and return_statistics:
            stats = metadata.get("statistics", {})
            stats_dict = {
                obj_type: Statistics(**stat_data) 
                for obj_type, stat_data in stats.items()
            }
            
            # Удаление временного файла
            if temp_file.exists():
                temp_file.unlink()
            
            return {
                "success": True,
                "message": f"Обработано объектов: {len(objects)}",
                "objects": objects,
                "statistics": stats_dict,
                "processing_time": metadata.get("processing_time"),
                "total_frames": metadata.get("total_frames"),
                "processed_frames": metadata.get("processed_frames"),
            }
        
        # Если нужно вернуть видео
        if return_video and output_path and os.path.exists(output_path):
            # Если также нужна статистика, возвращаем JSON с информацией и ссылкой на видео
            if return_statistics:
                stats = metadata.get("statistics", {})
                stats_dict = {
                    obj_type: Statistics(**stat_data) 
                    for obj_type, stat_data in stats.items()
                }
                
                # Удаление временного файла
                if temp_file.exists():
                    temp_file.unlink()
                
                return {
                    "success": True,
                    "message": f"Обработано объектов: {len(objects)}",
                    "objects": objects,
                    "statistics": stats_dict,
                    "processing_time": metadata.get("processing_time"),
                    "total_frames": metadata.get("total_frames"),
                    "processed_frames": metadata.get("processed_frames"),
                    "video_url": f"/download/{Path(output_path).name}",
                    "video_filename": Path(output_path).name
                }
            else:
                # Только видео без статистики
                # Удаление временного файла
                if temp_file.exists():
                    temp_file.unlink()
                
                return FileResponse(
                    path=output_path,
                    filename=Path(output_path).name,
                    media_type='video/mp4'
                )
        
        # Если видео не создано, возвращаем только статистику
        stats = metadata.get("statistics", {})
        stats_dict = {
            obj_type: Statistics(**stat_data) 
            for obj_type, stat_data in stats.items()
        } if return_statistics else None
        
        # Удаление временного файла
        if temp_file.exists():
            temp_file.unlink()
        
        return {
            "success": True,
            "message": f"Обработано объектов: {len(objects)}",
            "objects": objects,
            "statistics": stats_dict,
            "processing_time": metadata.get("processing_time"),
            "total_frames": metadata.get("total_frames"),
            "processed_frames": metadata.get("processed_frames"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Скачивание обработанного файла"""
    output_dir = Path(config.get("processing", {}).get("output_dir", "results"))
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


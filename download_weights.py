"""
Скрипт для автоматической загрузки весов и конфигурации YOLO v4
"""

import urllib.request
import os
from pathlib import Path


def download_file(url, filename):
    """Скачивание файла с прогресс-баром"""
    print(f"Скачивание {filename}...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 40
        filled = int(bar_length * downloaded / total_size)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print(f"\n{filename} успешно скачан!")
        return True
    except Exception as e:
        print(f"\nОшибка при скачивании {filename}: {e}")
        return False


def main():
    print("Загрузка файлов YOLO v4...")
    print("=" * 50)
    
    # Создаем директорию если нужно
    os.makedirs(".", exist_ok=True)
    
    # URL для скачивания
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
    
    weights_file = "yolov4.weights"
    config_file = "yolov4.cfg"
    
    # Проверка существующих файлов
    if os.path.exists(weights_file):
        response = input(f"Файл {weights_file} уже существует. Перезаписать? (y/n): ")
        if response.lower() != 'y':
            print(f"Пропуск {weights_file}")
        else:
            download_file(weights_url, weights_file)
    else:
        download_file(weights_url, weights_file)
    
    print()
    
    if os.path.exists(config_file):
        response = input(f"Файл {config_file} уже существует. Перезаписать? (y/n): ")
        if response.lower() != 'y':
            print(f"Пропуск {config_file}")
        else:
            download_file(config_url, config_file)
    else:
        download_file(config_url, config_file)
    
    print("\n" + "=" * 50)
    print("Загрузка завершена!")
    print("\nТеперь вы можете использовать:")
    print("python detect_video.py --input your_video.mp4")


if __name__ == "__main__":
    main()


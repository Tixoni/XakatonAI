"""
Модуль для загрузки конфигурации
"""

import os
import sys
import json


def load_config(config_path="config.json"):
    """Загрузка конфигурации"""
    # Получаем абсолютный путь к конфигу
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Поднимаемся на уровень выше (из src в корень)
        root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(root_dir, config_path)
    
    if not os.path.exists(config_path):
        print(f"Ошибка: файл config.json не найден!")
        print(f"Ожидался путь: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка при парсинге config.json: {e}")
        print(f"Проверьте синтаксис JSON файла")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при загрузке config.json: {e}")
        sys.exit(1)

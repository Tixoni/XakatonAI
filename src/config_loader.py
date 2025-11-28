import os
import sys
import json

def load_config(path="config.json"):
    """Загрузка конфигурации"""
    if not os.path.exists(path):
        print(f"Ошибка: файл {path} не найден!")
        sys.exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
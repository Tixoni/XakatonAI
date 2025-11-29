import os
import sys
import glob
from pathlib import Path
from datetime import datetime

def main():
    # Установка корневой директории
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Проверка существования директории
    if not os.path.exists(root_dir):
        print(f'Ошибка: Директория "{root_dir}" не существует!')
        input('Нажмите Enter для выхода...')
        return 1
    
    # Получение абсолютного пути
    root_dir = os.path.abspath(root_dir)
    
    print(f'Рабочая директория: {root_dir}')
    print()
    
    # Удалить существующий файл результатов
    if os.path.exists('Project_Files.txt'):
        os.remove('Project_Files.txt')
    
    # Список разрешенных расширений для исходного кода
    allowed_extensions = [
        '*.py', '*.json', '*.config','*.sql', '*.yml', '*.yaml', '.dockerignore', '.env.example'
    ]
    
    # Исключаемые директории
    exclude_dirs = ['.git', 'node_modules', 'bin', 'obj', 'packages', '.vs', '.idea']
    
    # Максимальный размер файла (1 МБ)
    max_size = 1048576
    
    print('Начало обработки файлов...')
    print()
    
    processed_files = 0
    skipped_files = 0
    
    # Рекурсивный поиск файлов
    for extension in allowed_extensions:
        pattern = os.path.join(root_dir, '**', extension)
        
        for file_path in glob.glob(pattern, recursive=True):
            # Проверка на исключаемые директории
            skip_file = False
            for exclude_dir in exclude_dirs:
                if exclude_dir in file_path.split(os.sep):
                    skip_file = True
                    break
            
            if skip_file:
                continue
            
            # Получение информации о файле
            try:
                file_size = os.path.getsize(file_path)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Проверка размера файла
                if file_size < max_size:
                    print(f'Обработка: {file_path}')
                    
                    with open('Project_Files.txt', 'a', encoding='utf-8') as result_file:
                        result_file.write(f'[Файл: {file_path}]\n')
                        result_file.write(f'[Размер: {file_size} байт]\n')
                        result_file.write(f'[Дата изменения: {mod_time}]\n')
                        result_file.write('\n')
                        
                        # Попытка чтения файла
                        try:
                            with open(file_path, 'r', encoding='utf-8') as src_file:
                                content = src_file.read()
                                result_file.write(content)
                        except UnicodeDecodeError:
                            # Попробуем другие кодировки
                            try:
                                with open(file_path, 'r', encoding='cp1251') as src_file:
                                    content = src_file.read()
                                    result_file.write(content)
                            except:
                                result_file.write('[Ошибка чтения файла - возможно бинарный файл]\n')
                        except Exception as e:
                            result_file.write(f'[Ошибка чтения файла: {str(e)}]\n')
                        
                        result_file.write('\n')
                        result_file.write('-----\n')
                        result_file.write('\n')
                    
                    processed_files += 1
                else:
                    print(f'Пропуск большого файла: {file_path} ({file_size} байт)')
                    
                    with open('Project_Files.txt', 'a', encoding='utf-8') as result_file:
                        result_file.write(f'[Файл: {file_path} - ПРОПУЩЕН (слишком большой: {file_size} байт)]\n')
                    
                    skipped_files += 1
                    
            except Exception as e:
                print(f'Ошибка при обработке файла {file_path}: {e}')
    
    print()
    print(f'Готово! Результат сохранен в Project_Files.txt')
    print(f'Обработано файлов: {processed_files}, пропущено: {skipped_files}')
    print(f'Обработана директория: {root_dir}')
    
    # Пауза в конце
    input('Нажмите Enter для выхода...')
    return 0

if __name__ == '__main__':
    sys.exit(main())
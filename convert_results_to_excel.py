#!/usr/bin/env python3
"""
Программа преобразования файла результатов эксперимента all_results.json
в формат для анализа в Excel (CSV/XLSX)
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime


def flatten_dict(d, parent_key='', sep='_'):
    """
    Рекурсивно выравнивает вложенный словарь в плоский словарь
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Если список содержит словари, обрабатываем каждый элемент
            if v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                # Преобразуем список в строку
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def convert_json_to_excel(input_file, output_format='csv'):
    """
    Конвертирует JSON файл в формат Excel (CSV или XLSX)
    
    Args:
        input_file: Путь к входному JSON файлу
        output_format: 'csv' или 'xlsx'
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Ошибка: Файл {input_file} не найден")
        sys.exit(1)
    
    # Чтение JSON файла
    print(f"Чтение файла {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка при чтении JSON: {e}")
            sys.exit(1)
    
    # Обработка данных
    # Предполагаем, что данные - это список сообщений/записей
    if isinstance(data, dict):
        # Если это один объект, превращаем в список
        if 'messages' in data or 'results' in data or 'experiments' in data:
            # Ищем ключ с данными
            for key in ['messages', 'results', 'experiments', 'data']:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
                elif key in data and isinstance(data[key], dict):
                    data = [data[key]]
                    break
        else:
            data = [data]
    
    if not isinstance(data, list):
        print("Ошибка: Данные должны быть списком записей или словарём с ключом 'messages'/'results'")
        sys.exit(1)
    
    if len(data) == 0:
        print("Предупреждение: Пустой набор данных")
        return
    
    print(f"Найдено записей: {len(data)}")
    
    # Выравнивание данных
    flattened_data = []
    for i, record in enumerate(data):
        if isinstance(record, dict):
            flat_record = flatten_dict(record)
            # Добавляем индекс записи если нужно
            flat_record['record_index'] = i
            flattened_data.append(flat_record)
        else:
            flattened_data.append({'value': record, 'record_index': i})
    
    # Определяем все возможные ключи
    all_keys = set()
    for record in flattened_data:
        all_keys.update(record.keys())
    
    # Сортируем ключи для удобного чтения
    sorted_keys = sorted(all_keys)
    
    # Генерация имени выходного файла
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = input_path.stem
    
    if output_format == 'csv':
        output_file = input_path.parent / f"{base_name}_export_{timestamp}.csv"
        
        # Запись в CSV
        print(f"Запись в CSV файл: {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerows(flattened_data)
        
        print(f"✓ Успешно экспортировано {len(flattened_data)} записей в {output_file}")
        
    elif output_format == 'xlsx':
        try:
            import pandas as pd
        except ImportError:
            print("Для формата XLSX требуется библиотека pandas. Установите: pip install pandas openpyxl")
            print("Переключаюсь на CSV формат...")
            output_format = 'csv'
            output_file = input_path.parent / f"{base_name}_export_{timestamp}.csv"
            
            print(f"Запись в CSV файл: {output_file}")
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=sorted_keys)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            print(f"✓ Успешно экспортировано {len(flattened_data)} записей в {output_file}")
            return
        
        output_file = input_path.parent / f"{base_name}_export_{timestamp}.xlsx"
        
        # Создание DataFrame и запись в XLSX
        print(f"Запись в XLSX файл: {output_file}")
        df = pd.DataFrame(flattened_data, columns=sorted_keys)
        df.to_excel(output_file, index=False, sheet_name='Results')
        
        print(f"✓ Успешно экспортировано {len(flattened_data)} записей в {output_file}")
    
    return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Преобразование all_results.json в формат Excel'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='all_results.json',
        help='Путь к входному JSON файлу (по умолчанию: all_results.json)'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['csv', 'xlsx'],
        default='csv',
        help='Формат вывода: csv или xlsx (по умолчанию: csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Конвертер JSON результатов в формат Excel")
    print("=" * 60)
    
    convert_json_to_excel(args.input_file, args.format)
    
    print("=" * 60)
    print("Готово!")


if __name__ == '__main__':
    main()

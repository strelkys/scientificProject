#!/usr/bin/env python3
"""
Скрипт для преобразования файла результатов эксперимента all_results.json
в формат Excel для дальнейшего анализа.

Распарсивает поле "metrics" на отдельные четыре столбца:
- metrics_0: первое числовое значение
- metrics_1: второе числовое значение
- metrics_2: третий элемент (массив) в виде строки
- metrics_3: четвертый элемент (массив) в виде строки
"""

import json
import pandas as pd
from pathlib import Path


def parse_metrics(metrics_list):
    """
    Распарсивает список metrics на 4 отдельных значения.
    
    Args:
        metrics_list: Список из 4 элементов [float, float, list, list]
    
    Returns:
        Кортеж из 4 значений (metrics_0, metrics_1, metrics_2, metrics_3)
    """
    if not metrics_list or len(metrics_list) < 4:
        return None, None, None, None
    
    # Первое значение - число
    metrics_0 = metrics_list[0]
    
    # Второе значение - число
    metrics_1 = metrics_list[1]
    
    # Третье значение - массив, преобразуем в строку
    metrics_2 = ", ".join(str(x) for x in metrics_list[2]) if isinstance(metrics_list[2], list) else str(metrics_list[2])
    
    # Четвертое значение - массив, преобразуем в строку
    metrics_3 = ", ".join(str(x) for x in metrics_list[3]) if isinstance(metrics_list[3], list) else str(metrics_list[3])
    
    return metrics_0, metrics_1, metrics_2, metrics_3


def convert_results_to_excel(input_json_path, output_excel_path):
    """
    Преобразует JSON файл с результатами экспериментов в Excel файл.
    
    Args:
        input_json_path: Путь к входному JSON файлу
        output_excel_path: Путь к выходному Excel файлу
    """
    # Чтение JSON файла
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Подготовка данных для DataFrame
    rows = []
    for record in data:
        row = {}
        
        # Копируем все поля кроме metrics
        for key, value in record.items():
            if key != 'metrics':
                row[key] = value
        
        # Распарсиваем metrics на 4 столбца
        if 'metrics' in record:
            m0, m1, m2, m3 = parse_metrics(record['metrics'])
            row['metrics_0'] = m0
            row['metrics_1'] = m1
            row['metrics_2'] = m2
            row['metrics_3'] = m3
        
        rows.append(row)
    
    # Создание DataFrame
    df = pd.DataFrame(rows)
    
    # Сохранение в Excel
    df.to_excel(output_excel_path, index=False, sheet_name='Results')
    
    print(f"Успешно преобразовано {len(rows)} записей")
    print(f"Результат сохранен в файл: {output_excel_path}")
    print(f"\nСтруктура данных:")
    print(df.columns.tolist())
    print(f"\nПример данных:")
    print(df.head())


if __name__ == "__main__":
    # Пути к файлам
    input_file = Path(__file__).parent / "all_results.json"
    output_file = Path(__file__).parent / "results.xlsx"
    
    # Проверка существования входного файла
    if not input_file.exists():
        print(f"Ошибка: Файл {input_file} не найден")
        exit(1)
    
    # Конвертация
    convert_results_to_excel(input_file, output_file)

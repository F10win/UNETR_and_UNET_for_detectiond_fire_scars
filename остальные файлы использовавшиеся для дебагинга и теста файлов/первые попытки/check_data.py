import sys
from pathlib import Path
import pandas as pd
import glob

# 1. ПРОВЕРКА ВЕРСИИ PYTHON
# PyTorch стабильно работает с Python 3.10, 3.11 или 3.12.
# Версия 3.14 пока не поддерживается большинством библиотек глубокого обучения.
current_version = sys.version_info
if current_version.major == 3 and current_version.minor >= 14:
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: Обнаружена неподдерживаемая версия Python!")
    print(f"   Ваша версия: Python {current_version.major}.{current_version.minor}")
    print("   Решение: Установите Python 3.11 или 3.12 и создайте новую виртуальную среду.")
    print("   Инструкция: https://www.python.org/downloads/")
    sys.exit(1)
else:
    print(f"✅ Версия Python: {current_version.major}.{current_version.minor} — подходит.")

print("📋 Проверка датасета\n")

# Настройки путей (замените на актуальные, если нужно)
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\raw_data")
CSV_PATH = BASE_DIR / "satellite_data.csv"
BASE_DIRS = [
    BASE_DIR / "Satellite_burned_area_dataset_part1",
    BASE_DIR / "Satellite_burned_area_dataset_part2",
    BASE_DIR / "Satellite_burned_area_dataset_part3",
    BASE_DIR / "Satellite_burned_area_dataset_part4",
    BASE_DIR / "Satellite_burned_area_dataset_part5",
]

# Читаем CSV с правильным разделителем (точка с запятой)
try:
    df = pd.read_csv(CSV_PATH, sep=';')
    print(f"✅ В CSV найдено {len(df)} записей")
    print(f"📊 Уникальные fold'ы: {df['fold'].unique()}")
    print(f"📊 Распределение:\n{df['fold'].value_counts()}\n")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл {CSV_PATH} не найден. Проверьте путь.")
except Exception as e:
    print(f"❌ Ошибка при чтении CSV: {e}")

# Проверяем папки
for base_dir in BASE_DIRS:
    if base_dir.exists():
        folders = list(base_dir.iterdir())
        print(f"✅ {base_dir.name}: {len(folders)} папок")
    else:
        print(f"❌ {base_dir.name}: НЕ НАЙДЕНА")

print("\n📁 Пример структуры:")
sample_folder = BASE_DIRS[0]
if sample_folder.exists():
    folders = list(sample_folder.iterdir())
    if folders:
        first_folder = folders[0]
        print(f"\n{first_folder.name}:")
        files = list(first_folder.iterdir())
        for f in files[:5]:  # Первые 5 файлов
            print(f"  - {f.name}")
        if len(files) > 5:
            print(f"  ... и ещё {len(files) - 5} файлов")
    else:
        print("  (Папка пуста)")

print("\n✅ Проверка завершена!")
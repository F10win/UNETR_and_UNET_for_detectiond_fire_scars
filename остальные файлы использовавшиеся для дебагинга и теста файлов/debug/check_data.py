from pathlib import Path
import pandas as pd
import glob

BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
CSV_PATH = BASE_DIR / "satellite_data.csv"

BASE_DIRS = [
    BASE_DIR / "Satellite_burned_area_dataset_part1",
    BASE_DIR / "Satellite_burned_area_dataset_part2",
    BASE_DIR / "Satellite_burned_area_dataset_part3",
    BASE_DIR / "Satellite_burned_area_dataset_part4",
    BASE_DIR / "Satellite_burned_area_dataset_part5",
]

print("📋 Проверка датасета\n")

# Читаем CSV
df = pd.read_csv(CSV_PATH, sep=';')
print(f"✅ В CSV найдено {len(df)} записей")
print(f"📊 Уникальные fold'ы: {df['fold'].unique()}")
print(f"📊 Распределение:\n{df['fold'].value_counts()}\n")

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
    first_folder = list(sample_folder.iterdir())[0]
    print(f"\n{first_folder.name}:")
    files = list(first_folder.iterdir())
    for f in files[:5]:  # Первые 5 файлов
        print(f"  - {f.name}")
    if len(files) > 5:
        print(f"  ... и ещё {len(files) - 5} файлов")
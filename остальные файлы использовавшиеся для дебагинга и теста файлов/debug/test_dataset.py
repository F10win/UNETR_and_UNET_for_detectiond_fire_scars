from pathlib import Path
from dataset import SatelliteBurnDataset
import os

print("🧪 Тестирование dataset.py\n")

# Получаем текущую директорию скрипта
CURRENT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
print(f"📂 Текущая директория: {CURRENT_DIR}\n")

# Пробуем разные варианты путей
POSSIBLE_PATHS = [
    CURRENT_DIR / "satellite_data.csv",
    Path(r"E:\миигаик\1_Магистратура\Диплом\проект\Satellite Burned Area Dataset\satellite_data.csv"),
    Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\satellite_data.csv"),
]

CSV_PATH = None
for path in POSSIBLE_PATHS:
    if path.exists():
        CSV_PATH = path
        print(f"✅ CSV найден: {CSV_PATH}\n")
        break

if CSV_PATH is None:
    print("❌ CSV файл не найден!\n")
    print("Искал в следующих местах:")
    for path in POSSIBLE_PATHS:
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {path}")
    print("\n💡 Укажи правильный путь к satellite_data.csv вручную:")
    raise FileNotFoundError("satellite_data.csv не найден")

BASE_DIR = CSV_PATH.parent

BASE_DIRS = [
    BASE_DIR / "Satellite_burned_area_dataset_part1",
    BASE_DIR / "Satellite_burned_area_dataset_part2",
    BASE_DIR / "Satellite_burned_area_dataset_part3",
    BASE_DIR / "Satellite_burned_area_dataset_part4",
    BASE_DIR / "Satellite_burned_area_dataset_part5",
]

# Проверяем существование папок
print("📁 Проверка папок с данными:")
for base_dir in BASE_DIRS:
    exists = "✅" if base_dir.exists() else "❌"
    print(f"  {exists} {base_dir.name}")
print()

# Тестируем с одним fold'ом
test_folds = ['purple']

print(f"📂 Создание датасета для fold'ов: {test_folds}\n")

try:
    dataset = SatelliteBurnDataset(
        base_dirs=BASE_DIRS,
        csv_path=CSV_PATH,
        fold_colors=test_folds,
        img_size=(512, 512)
    )
    
    print(f"\n✅ Датасет создан!")
    print(f"📊 Размер датасета: {len(dataset)} примеров\n")
    
    # Пробуем загрузить первый пример
    print("🔄 Загрузка первого примера...")
    img, mask = dataset[0]
    
    print(f"\n✅ Успешно!")
    print(f"📐 Размер изображения: {img.shape}")
    print(f"📐 Размер маски: {mask.shape}")
    print(f"🔢 Количество каналов: {img.shape[0]}")
    print(f"📊 Min/Max значения маски: {mask.min().item():.2f} / {mask.max().item():.2f}")
    print(f"📊 Min/Max значения изображения: {img.min().item():.3f} / {img.max().item():.3f}")
    
    # Проверяем процент горящих пикселей
    burned_pixels = (mask > 0.5).sum().item()
    total_pixels = mask.numel()
    burned_percent = (burned_pixels / total_pixels) * 100
    print(f"🔥 Горящих пикселей: {burned_percent:.2f}%")
    
    print("\n✅ Все тесты пройдены!")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
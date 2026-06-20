from pathlib import Path
import tifffile
import numpy as np

print("🔍 Диагностика TIFF файлов\n")

BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
FIRST_PART = BASE_DIR / "Satellite_burned_area_dataset_part1"

# Получаем первую папку
folders = list(FIRST_PART.iterdir())
first_folder = folders[0]

print(f"📁 Папка: {first_folder.name}\n")

# Проверяем Sentinel-2 файлы
s2_files = list(first_folder.glob("sentinel2_*.tiff"))

for s2_file in s2_files:
    print(f"📄 Файл: {s2_file.name}")
    try:
        # Читаем информацию о TIFF
        with tifffile.TiffFile(s2_file) as tif:
            print(f"  Количество страниц: {len(tif.pages)}")
            
            # Читаем первое изображение
            img = tifffile.imread(s2_file)
            print(f"  Форма массива: {img.shape}")
            print(f"  Тип данных: {img.dtype}")
            print(f"  Min/Max: {img.min()} / {img.max()}")
            
            if len(img.shape) == 3:
                if img.shape[0] < img.shape[-1]:  # Каналы первые
                    print(f"  📊 Каналов: {img.shape[0]} (формат: C,H,W)")
                else:  # Каналы последние
                    print(f"  📊 Каналов: {img.shape[-1]} (формат: H,W,C)")
            else:
                print(f"  📊 2D изображение (один канал)")
                
    except Exception as e:
        print(f"  ❌ Ошибка чтения: {e}")
    print()

# Проверяем маску
mask_files = list(first_folder.glob("*_mask.tiff"))
if mask_files:
    mask_file = mask_files[0]
    print(f"📄 Маска: {mask_file.name}")
    mask = tifffile.imread(mask_file)
    print(f"  Форма: {mask.shape}")
    print(f"  Тип: {mask.dtype}")
    print(f"  Уникальные значения: {np.unique(mask)[:10]}...")  # Первые 10
    print(f"  Min/Max: {mask.min()} / {mask.max()}")
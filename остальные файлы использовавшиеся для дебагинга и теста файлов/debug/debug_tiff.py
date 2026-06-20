from pathlib import Path
import tifffile
import numpy as np

BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
FIRST_PART = BASE_DIR / "Satellite_burned_area_dataset_part1"

# Получаем первую папку
folders = list(FIRST_PART.iterdir())
first_folder = folders[0]

print(f"📁 Папка: {first_folder.name}\n")

# Ищем Sentinel-2 файлы
s2_files = list(first_folder.glob("sentinel2_*.tiff"))

for s2_file in s2_files[:1]:  # Берём один файл для теста
    print(f"📄 Файл: {s2_file.name}")
    
    # Читаем информацию
    with tifffile.TiffFile(s2_file) as tif:
        print(f"  Количество страниц: {len(tif.pages)}")
        
        # Читаем данные
        img = tifffile.imread(s2_file)
        print(f"  Форма: {img.shape}")
        print(f"  Тип данных: {img.dtype}")
        print(f"  Min/Max: {img.min()} / {img.max()}")
        
        if len(img.shape) == 3:
            if img.shape[0] < img.shape[-1]:
                print(f"  📊 Каналы первые: {img.shape[0]} каналов")
            else:
                print(f"  📊 Каналы последние: {img.shape[-1]} каналов")
        else:
            print(f"  📊 2D изображение (1 канал)")
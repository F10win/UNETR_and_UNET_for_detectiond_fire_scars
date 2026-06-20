from pathlib import Path
import glob

print("🔍 Диагностика структуры файлов\n")

BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
FIRST_PART = BASE_DIR / "Satellite_burned_area_dataset_part1"

# Получаем первую папку с данными
folders = list(FIRST_PART.iterdir())
if not folders:
    print("❌ Папки пусты!")
else:
    first_folder = folders[0]
    print(f"📁 Первая папка: {first_folder.name}\n")
    
    # Показываем ВСЕ файлы
    all_files = list(first_folder.iterdir())
    print(f"📄 Все файлы ({len(all_files)}):")
    for f in all_files:
        print(f"  - {f.name}")
    
    print("\n" + "="*60)
    
    # Пробуем разные шаблоны
    patterns = [
        "sentinel2_*.tiff",
        "sentinel2_*.tif",
        "sentinel2_*.png",
        "Sentinel2_*.tiff",
        "S2*.tiff",
        "*.tiff",
        "*.tif",
        "*sentinel*.tiff",
    ]
    
    print("\n🔍 Поиск по шаблонам:")
    for pattern in patterns:
        files = list(glob.glob(str(first_folder / pattern)))
        if files:
            print(f"  ✅ '{pattern}': {len(files)} файлов")
            for f in files[:3]:  # Первые 3
                print(f"      - {Path(f).name}")
        else:
            print(f"  ❌ '{pattern}': не найдено")
    
    print("\n" + "="*60)
    
    # Проверяем маски
    print("\n📋 Маски:")
    mask_patterns = ["*_mask.tiff", "*_mask.png", "*mask*.tiff"]
    for pattern in mask_patterns:
        files = list(glob.glob(str(first_folder / pattern)))
        if files:
            print(f"  ✅ '{pattern}': {len(files)} файлов")
            for f in files[:2]:
                print(f"      - {Path(f).name}")
import rasterio
from rasterio.merge import merge
from pathlib import Path
import sys

def merge_geotiffs(input_files: list[str], output_file: str):
    """
    Объединяет несколько GeoTIFF файлов в один с сохранением геопривязки и метаданных.
    """
    if not input_files:
        raise ValueError("Список входных файлов пуст.")
    
    # Открываем все файлы
    print(f"📂 Открываю {len(input_files)} файлов...")
    src_files = [rasterio.open(str(f)) for f in input_files]
    
    # Выполняем слияние
    print("🔄 Выполняю слияние (merge)...")
    mosaic, out_trans = merge(src_files)
    
    # Копируем метаданные первого файла и обновляем геометрию
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]  # Количество каналов
    })
    
    # Записываем результат
    print(f"💾 Сохраняю результат в {output_file}...")
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    # Закрываем исходные файлы
    for src in src_files:
        src.close()
        
    print(f"✅ Успешно! Объединено {len(input_files)} файлов → {output_file}")

if __name__ == "__main__":
    # === НАСТРОЙКИ ===
    # Укажите пути к частям, экспортированным из GEE
    PART_1 = "E:\\миигаик\\1_Магистратура\\Диплом\\проект\\UNETR_Burn\\Ilusha\\raw_data\\region_05\\20210725T042709_20210725T042705_T49WDP_11ch-0000000000-0000000000.tif"
    PART_2 = "E:\\миигаик\\1_Магистратура\\Диплом\\проект\\UNETR_Burn\\Ilusha\\raw_data\\region_05\\20210725T042709_20210725T042705_T49WDP_11ch-0000000000-0000014080.tif"
    # Если частей больше, просто добавьте их в список
    INPUT_FILES = [PART_1, PART_2]
    
    OUTPUT_FILE = "merged_output.tif"
    # =================
    
    try:
        merge_geotiffs(INPUT_FILES, OUTPUT_FILE)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
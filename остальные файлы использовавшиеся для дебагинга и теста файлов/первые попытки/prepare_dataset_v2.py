import os
import random
import numpy as np
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
import fiona
from fiona.transform import transform_geom
import shutil
from tqdm import tqdm
import pyproj

# ============================================================================
# ⚙️ НАСТРОЙКИ (МЕНЯЙ ЭТИ ПЕРЕМЕННЫЕ ПЕРЕД КАЖДЫМ РЕГИОНОМ)
# ============================================================================

# Путь к папке региона
REGION_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\raw_data\region_15")

# Начальный номер патча (для продолжения нумерации)
START_PATCH = 6086

# Размер патча и overlap
PATCH_SIZE = 512
OVERLAP = 256

# 🆕 Процент сохранения "пустых" патчей (где 0% гарь)
EMPTY_PATCHES_SAVE_PERCENT = 75

# Выходная папка датасета
OUTPUT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset")

# ============================================================================

def find_files(folder):
    """Автоматический поиск файлов по суффиксам"""
    print(f"\n🔍 Поиск файлов в {folder}...")
    print("="*70)
    
    files = {
        'after_11ch': None,
        'after_3ch': None,
        'shp': None
    }
    
    tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    shp_files = list(folder.glob("*.shp"))
    
    print(f"   Найдено TIFF файлов: {len(tiff_files)}")
    print(f"   Найдено SHP файлов: {len(shp_files)}")
    
    for f in tiff_files:
        name = f.name.lower()
        if '_13chan_after' in name:
            files['after_11ch'] = f
            print(f"   ✅ after_11ch (13chan): {f.name}")
        elif '_3chan_after' in name:
            files['after_3ch'] = f
            print(f"   ✅ after_3ch: {f.name}")
    
    for f in shp_files:
        if '_fire_boundary' in f.name.lower():
            files['shp'] = f
            print(f"   ✅ SHP (mask): {f.name}")
    
    return files

def rasterize_shp(shp_path, reference_tiff, output_mask_path):
    """Растрирование SHP файла в бинарную маску с перепроецированием"""
    print(f"\n🔄 Растрирование SHP: {shp_path.name}")
    
    with rasterio.open(reference_tiff) as src:
        transform = src.transform
        shape_img = src.shape
        target_crs = src.crs
    
    print(f"   📐 Размер: {shape_img[1]}×{shape_img[0]}")
    print(f"   📐 Целевая проекция (TIFF): {target_crs}")
    
    geometries = []
    source_crs = None
    
    with fiona.open(shp_path, 'r') as shp:
        source_crs = shp.crs
        print(f"   📐 Исходная проекция (SHP): {source_crs}")
        
        if source_crs != target_crs:
            print(f"   ⚠️  CRS не совпадают! Выполняется перепроецирование...")
            print(f"   🔄 {source_crs} -> {target_crs}")
            
            for feature in shp:
                # Перепроецируем геометрию
                transformed_geom = transform_geom(
                    source_crs, 
                    target_crs, 
                    feature['geometry']
                )
                geom = shape(transformed_geom)
                geometries.append(geom)
        else:
            print(f"   ✅ CRS совпадают, перепроецирование не требуется")
            for feature in shp:
                geom = shape(feature['geometry'])
                geometries.append(geom)
    
    print(f"   📐 Найдено полигонов: {len(geometries)}")
    
    # Растеризация
    mask = rasterize(
        geometries,
        out_shape=shape_img,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )
    
    fire_pixels = (mask > 0).sum()
    total_pixels = mask.size
    fire_percent = (fire_pixels / total_pixels) * 100
    
    print(f"   🔥 Пожарных пикселей: {fire_pixels} / {total_pixels} ({fire_percent:.2f}%)")
    
    with rasterio.open(
        output_mask_path,
        'w',
        driver='GTiff',
        height=shape_img[0],
        width=shape_img[1],
        count=1,
        dtype=np.uint8,
        crs=target_crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(mask, 1)
    
    print(f"   ✅ Маска сохранена: {output_mask_path.name}")
    return mask

def create_patches(image_11ch, image_3ch, mask, region_name, output_dir_11ch, output_dir_3ch, 
                   transform_11ch=None, transform_3ch=None, crs=None, start_patch=1):
    """Нарезка на патчи 512×512 с overlap, геопривязкой и фильтрацией фона"""
    print(f"\n Нарезка на патчи {PATCH_SIZE}×{PATCH_SIZE} (overlap={OVERLAP}px)...")
    print(f"   📋 Начальный номер патча: {start_patch}")
    print(f"   🎛️  Доля сохранения фона: {EMPTY_PATCHES_SAVE_PERCENT}%")
    
    save_pct = max(0, min(100, EMPTY_PATCHES_SAVE_PERCENT))
    
    if image_11ch.ndim == 3 and image_11ch.shape[0] < 20:
        c, h, w = image_11ch.shape
    else:
        image_11ch = np.transpose(image_11ch, (2, 0, 1))
        image_3ch = np.transpose(image_3ch, (2, 0, 1))
        c, h, w = image_11ch.shape
    
    print(f"   📐 Размер изображения: {w}×{h} px, {c} каналов")
    
    saved_count = 0
    saved_background = 0
    saved_fire = 0
    current_patch_num = start_patch
    
    step_y = PATCH_SIZE - OVERLAP
    step_x = PATCH_SIZE - OVERLAP
    
    if h < PATCH_SIZE or w < PATCH_SIZE:
        print(f"   ⚠️  Изображение слишком маленькое ({w}×{h}) для патчей {PATCH_SIZE}×{PATCH_SIZE}")
        return 0, start_patch
    
    total_steps_x = (w - PATCH_SIZE) // step_x + 1
    total_steps_y = (h - PATCH_SIZE) // step_y + 1
    total_patches = total_steps_x * total_steps_y
    
    with tqdm(total=total_patches, desc="Нарезка", unit="patch") as pbar:
        for y in range(0, h - PATCH_SIZE + 1, step_y):
            for x in range(0, w - PATCH_SIZE + 1, step_x):
                pbar.update(1)
                
                patch_11ch = image_11ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_3ch = image_3ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_mask = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                fire_pixels = (patch_mask > 0).sum()
                total_pixels = patch_mask.size
                fire_percent = (fire_pixels / total_pixels) * 100
                
                if fire_percent > 0:
                    saved_fire += 1
                    keep_patch = True
                else:
                    if random.random() * 100 < save_pct:
                        saved_background += 1
                        keep_patch = True
                    else:
                        keep_patch = False
                
                if not keep_patch:
                    continue
                
                saved_count += 1
                patch_name = f"patch_{current_patch_num:04d}"
                current_patch_num += 1
                
                patch_dir_11ch = output_dir_11ch / patch_name
                patch_dir_3ch = output_dir_3ch / patch_name
                patch_dir_11ch.mkdir(exist_ok=True)
                patch_dir_3ch.mkdir(exist_ok=True)
                
                patch_transform_11ch = None
                if transform_11ch is not None:
                    patch_transform_11ch = Affine(
                        transform_11ch.a, transform_11ch.b,
                        transform_11ch.c + x * transform_11ch.a,
                        transform_11ch.d, transform_11ch.e,
                        transform_11ch.f + y * transform_11ch.e
                    )
                
                patch_transform_3ch = None
                if transform_3ch is not None:
                    patch_transform_3ch = Affine(
                        transform_3ch.a, transform_3ch.b,
                        transform_3ch.c + x * transform_3ch.a,
                        transform_3ch.d, transform_3ch.e,
                        transform_3ch.f + y * transform_3ch.e
                    )
                
                with rasterio.open(
                    patch_dir_11ch / "after_11ch.tif", 'w', driver='GTiff',
                    height=PATCH_SIZE, width=PATCH_SIZE,
                    count=patch_11ch.shape[0], dtype=patch_11ch.dtype,
                    crs=crs, transform=patch_transform_11ch,
                    compress='lzw', photometric='minisblack'
                ) as dst:
                    dst.write(patch_11ch)
                
                with rasterio.open(
                    patch_dir_3ch / "after_3ch.tif", 'w', driver='GTiff',
                    height=PATCH_SIZE, width=PATCH_SIZE,
                    count=patch_3ch.shape[0], dtype=patch_3ch.dtype,
                    crs=crs, transform=patch_transform_3ch,
                    compress='lzw', photometric='rgb'
                ) as dst:
                    dst.write(patch_3ch)
                
                with rasterio.open(
                    patch_dir_11ch / "mask.tif", 'w', driver='GTiff',
                    height=PATCH_SIZE, width=PATCH_SIZE, count=1, dtype=np.uint8,
                    crs=crs, transform=patch_transform_11ch or patch_transform_3ch,
                    compress='lzw', photometric='minisblack'
                ) as dst:
                    dst.write(patch_mask.astype(np.uint8), 1)
                
                shutil.copy(patch_dir_11ch / "mask.tif", patch_dir_3ch / "mask.tif")
    
    print(f"   ✅ Создано патчей: {saved_count} (с {start_patch} по {current_patch_num - 1})")
    print(f"   🔥 С гарью: {saved_fire} (100% сохранено)")
    print(f"   🌿 Фоновых: {saved_background} (отфильтровано до {save_pct}%)")
    
    return saved_count, current_patch_num

def process_region(region_path, start_patch):
    """Обработка одного региона"""
    print("="*70)
    print(f"📁 Обработка региона: {region_path.name}")
    print("="*70)
    
    files = find_files(region_path)
    
    required = ['after_11ch', 'after_3ch', 'shp']
    missing = [f for f in required if files[f] is None]
    
    if missing:
        print(f"\n❌ Отсутствуют файлы:")
        for f in missing:
            print(f"   - {f}")
        return 0, start_patch
    
    print(f"\n✅ Все файлы найдены!")
    print("\n📷 Загрузка TIFF файлов...")
    
    transform_11ch = None
    transform_3ch = None
    crs = None
    
    try:
        print(f"   📄 {files['after_11ch'].name}")
        with rasterio.open(files['after_11ch']) as src:
            after_11ch = src.read().astype(np.float32)
            transform_11ch = src.transform
            crs = src.crs
            print(f"      Каналы: {after_11ch.shape[0]}, Размер: {after_11ch.shape[2]}×{after_11ch.shape[1]}")
        
        print(f"   📄 {files['after_3ch'].name}")
        with rasterio.open(files['after_3ch']) as src:
            after_3ch = src.read().astype(np.float32)
            transform_3ch = src.transform
            print(f"      Каналы: {after_3ch.shape[0]}, Размер: {after_3ch.shape[2]}×{after_3ch.shape[1]}")
        
        print("   ✅ Все TIFF загружены с геопривязкой")
        
    except Exception as e:
        print(f"   ❌ Ошибка при загрузке TIFF: {e}")
        import traceback
        traceback.print_exc()
        return 0, start_patch
    
    mask_path = region_path / "mask_temp.tif"
    try:
        mask = rasterize_shp(files['shp'], files['after_11ch'], mask_path)
    except Exception as e:
        print(f"   ❌ Ошибка при растрировании: {e}")
        import traceback
        traceback.print_exc()
        return 0, start_patch
    
    output_dir_11ch = OUTPUT_DIR / "13_channels"
    output_dir_3ch = OUTPUT_DIR / "3_channels"
    output_dir_11ch.mkdir(parents=True, exist_ok=True)
    output_dir_3ch.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Создание патчей...")
    patch_count, next_patch_num = create_patches(
        after_11ch, after_3ch, mask,
        region_path.name,
        output_dir_11ch, output_dir_3ch,
        transform_11ch, transform_3ch, crs,
        start_patch
    )
    
    if mask_path.exists():
        mask_path.unlink()
    
    return patch_count, next_patch_num

def main():
    print("="*70)
    print("🛰️  Подготовка датасета Satellite Burned Area (UNIVERSAL)")
    print("="*70)
    print(f"\n⚙️  НАСТРОЙКИ:")
    print(f"   📂 Регион: {REGION_PATH}")
    print(f"   🔢 Начальный патч: {START_PATCH}")
    print(f"   📐 Размер патча: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"   📐 Overlap: {OVERLAP} px")
    print(f"   🌿 % Фона к сохранению: {EMPTY_PATCHES_SAVE_PERCENT}%")
    print(f"   📁 Выход: {OUTPUT_DIR}")
    print("="*70)
    
    if not REGION_PATH.exists():
        print(f"\n❌ Папка региона не найдена: {REGION_PATH}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    patch_count, next_patch = process_region(REGION_PATH, START_PATCH)
    
    print(f"\n{'='*70}")
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print("="*70)
    
    if patch_count > 0:
        print(f"✅ Регион: {REGION_PATH.name}")
        print(f"📊 Создано патчей: {patch_count}")
        print(f"🔢 Следующий начальный номер: {next_patch}")
        print(f"\n💡 Для обработки следующего региона:")
        print(f"   1. Измени REGION_PATH на путь к следующему региону")
        print(f"   2. Установи START_PATCH = {next_patch}")
        print(f"   3. (Опционально) Измени EMPTY_PATCHES_SAVE_PERCENT")
        print(f"   4. Запусти скрипт снова")
    else:
        print("⚠️  Патчи не созданы!")
        print("   Возможные причины:")
        print("   1. Изображение слишком маленькое")
        print("   2. В маске нет пожаров")
        print("   3. Пожары занимают < 1% площади")
    
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
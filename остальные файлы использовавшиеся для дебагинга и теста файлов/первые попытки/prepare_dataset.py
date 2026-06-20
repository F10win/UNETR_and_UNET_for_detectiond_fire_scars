import os
import numpy as np
import tifffile
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import fiona
import shutil
from tqdm import tqdm

# ============================================================================
# ⚙️ НАСТРОЙКИ (МЕНЯЙ ЭТИ ПЕРЕМЕННЫЕ ПЕРЕД КАЖДЫМ РЕГИОНОМ)
# ============================================================================

# Путь к папке региона (меняй на region_01, region_02 и т.д.)
REGION_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\raw_data\region_03")

# Начальный номер патча (для продолжения нумерации)
# Если начинаешь с нуля: START_PATCH = 1
# Если уже есть 270 патчей: START_PATCH = 271
START_PATCH = 706

# Размер патча и overlap
PATCH_SIZE = 512
OVERLAP = 256

# Минимальный процент пожара в патче (иначе отбрасываем)
MIN_FIRE_PERCENT = 0

# Выходная папка датасета
OUTPUT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset")

# ============================================================================


def find_files(folder):
    """Автоматический поиск файлов по суффиксам"""
    print(f"\n🔍 Поиск файлов в {folder}...")
    print("="*70)
    
    files = {
        'before_13ch': None,
        'after_13ch': None,
        'before_3ch': None,
        'after_3ch': None,
        'shp': None
    }
    
    # Ищем все TIFF и SHP файлы
    tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    shp_files = list(folder.glob("*.shp"))
    
    print(f"   Найдено TIFF файлов: {len(tiff_files)}")
    print(f"   Найдено SHP файлов: {len(shp_files)}")
    
    for f in tiff_files:
        name = f.name.lower()
        
        # Определяем тип файла по суффиксу
        if '_13chan_before' in name or '_13chan_before.' in name:
            files['before_13ch'] = f
            print(f"   ✅ before_13ch: {f.name}")
        elif '_13chan_after' in name or '_13chan_after.' in name:
            files['after_13ch'] = f
            print(f"   ✅ after_13ch: {f.name}")
        elif '_3chan_before' in name or '_3chan_before.' in name:
            files['before_3ch'] = f
            print(f"   ✅ before_3ch: {f.name}")
        elif '_3chan_after' in name or '_3chan_after.' in name:
            files['after_3ch'] = f
            print(f"   ✅ after_3ch: {f.name}")
    
    for f in shp_files:
        if '_fire_boundary' in f.name.lower():
            files['shp'] = f
            print(f"   ✅ SHP: {f.name}")
    
    return files

def rasterize_shp(shp_path, reference_tiff, output_mask_path):
    """Растрирование SHP файла в бинарную маску"""
    print(f"\n🔄 Растрирование SHP: {shp_path.name}")
    
    with rasterio.open(reference_tiff) as src:
        transform = src.transform
        shape_img = src.shape
        crs = src.crs
    
    print(f"   📐 Размер: {shape_img[1]}×{shape_img[0]}")
    print(f"   📐 Проекция: {crs}")
    
    with fiona.open(shp_path, 'r') as shp:
        geometries = []
        for feature in shp:
            geom = shape(feature['geometry'])
            geometries.append(geom)
        print(f"   📐 Найдено полигонов: {len(geometries)}")
    
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
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(mask, 1)
    
    print(f"   ✅ Маска сохранена: {output_mask_path.name}")
    return mask

def create_patches(image_13ch, image_3ch, mask, region_name, output_dir_13ch, output_dir_3ch, 
                   before_13ch=None, before_3ch=None, 
                   transform_13ch=None, transform_3ch=None, crs=None,
                   start_patch=1):
    """Нарезка на патчи 512×512 с overlap, геопривязкой и нумерацией"""
    print(f"\n🔄 Нарезка на патчи {PATCH_SIZE}×{PATCH_SIZE} (overlap={OVERLAP}px)...")
    print(f"   📋 Начальный номер патча: {start_patch}")
    
    # Определяем размеры и формат
    if image_13ch.ndim == 3 and image_13ch.shape[0] < 20:
        c, h, w = image_13ch.shape
        channels_first = True
    else:
        image_13ch = np.transpose(image_13ch, (2, 0, 1))
        image_3ch = np.transpose(image_3ch, (2, 0, 1))
        if before_13ch is not None:
            before_13ch = np.transpose(before_13ch, (2, 0, 1))
        if before_3ch is not None:
            before_3ch = np.transpose(before_3ch, (2, 0, 1))
        c, h, w = image_13ch.shape
        channels_first = True
    
    print(f"   📐 Размер изображения: {w}×{h} px, {c} каналов")
    
    patch_count = 0
    saved_count = 0
    skipped_no_fire = 0
    current_patch_num = start_patch
    
    step_y = PATCH_SIZE - OVERLAP
    step_x = PATCH_SIZE - OVERLAP
    
    if h < PATCH_SIZE or w < PATCH_SIZE:
        print(f"   ⚠️  Изображение слишком маленькое ({w}×{h}) для патчей {PATCH_SIZE}×{PATCH_SIZE}")
        return 0, start_patch
    
    for y in range(0, h - PATCH_SIZE + 1, step_y):
        for x in range(0, w - PATCH_SIZE + 1, step_x):
            patch_count += 1
            
            patch_13ch = image_13ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_3ch = image_3ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_mask = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            fire_pixels = (patch_mask > 0).sum()
            total_pixels = patch_mask.size
            fire_percent = (fire_pixels / total_pixels) * 100
            
            #if fire_percent < MIN_FIRE_PERCENT:
            #    skipped_no_fire += 1
            #    continue
            
            saved_count += 1
            patch_name = f"patch_{current_patch_num:04d}"
            current_patch_num += 1
            
            patch_dir_13ch = output_dir_13ch / patch_name
            patch_dir_3ch = output_dir_3ch / patch_name
            patch_dir_13ch.mkdir(exist_ok=True)
            patch_dir_3ch.mkdir(exist_ok=True)
            
            # === 13 КАНАЛОВ - AFTER ===
            patch_13ch_path = patch_dir_13ch / "after_13ch.tif"
            if transform_13ch is not None:
                patch_transform_13ch = rasterio.transform.Affine(
                    transform_13ch.a, transform_13ch.b,
                    transform_13ch.c + x * transform_13ch.a,
                    transform_13ch.d, transform_13ch.e,
                    transform_13ch.f + y * transform_13ch.e
                )
            else:
                patch_transform_13ch = None
            
            with rasterio.open(
                patch_13ch_path, 'w', driver='GTiff',
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=patch_13ch.shape[0],
                dtype=patch_13ch.dtype,
                crs=crs, transform=patch_transform_13ch,
                compress='lzw', photometric='minisblack'
            ) as dst:
                dst.write(patch_13ch)
            
            # === 3 КАНАЛА - AFTER ===
            patch_3ch_path = patch_dir_3ch / "after_3ch.tif"
            if transform_3ch is not None:
                patch_transform_3ch = rasterio.transform.Affine(
                    transform_3ch.a, transform_3ch.b,
                    transform_3ch.c + x * transform_3ch.a,
                    transform_3ch.d, transform_3ch.e,
                    transform_3ch.f + y * transform_3ch.e
                )
            else:
                patch_transform_3ch = None
            
            with rasterio.open(
                patch_3ch_path, 'w', driver='GTiff',
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=patch_3ch.shape[0],
                dtype=patch_3ch.dtype,
                crs=crs, transform=patch_transform_3ch,
                compress='lzw', photometric='rgb'
            ) as dst:
                dst.write(patch_3ch)
            
            # === МАСКА ===
            mask_path = patch_dir_13ch / "mask.tif"
            with rasterio.open(
                mask_path, 'w', driver='GTiff',
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=1, dtype=np.uint8,
                crs=crs,
                transform=patch_transform_13ch if patch_transform_13ch else patch_transform_3ch,
                compress='lzw', photometric='minisblack'
            ) as dst:
                dst.write(patch_mask.astype(np.uint8), 1)
            
            mask_path_3ch = patch_dir_3ch / "mask.tif"
            shutil.copy(mask_path, mask_path_3ch)
            
            # === BEFORE ПАТЧИ ===
            if before_13ch is not None:
                before_patch_13ch = before_13ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                before_patch_3ch = before_3ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            else:
                before_patch_13ch = np.zeros_like(patch_13ch)
                before_patch_3ch = np.zeros_like(patch_3ch)
            
            before_13ch_path = patch_dir_13ch / "before_13ch.tif"
            with rasterio.open(
                before_13ch_path, 'w', driver='GTiff',
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=before_patch_13ch.shape[0],
                dtype=before_patch_13ch.dtype,
                crs=crs, transform=patch_transform_13ch,
                compress='lzw', photometric='minisblack'
            ) as dst:
                dst.write(before_patch_13ch)
            
            before_3ch_path = patch_dir_3ch / "before_3ch.tif"
            with rasterio.open(
                before_3ch_path, 'w', driver='GTiff',
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=before_patch_3ch.shape[0],
                dtype=before_patch_3ch.dtype,
                crs=crs, transform=patch_transform_3ch,
                compress='lzw', photometric='rgb'
            ) as dst:
                dst.write(before_patch_3ch)
    
    print(f"   ✅ Создано патчей: {saved_count} (с {start_patch} по {current_patch_num - 1})")
    print(f"   ⏭️  Пропущено (нет пожара): {skipped_no_fire}")
    
    return saved_count, current_patch_num

def process_region(region_path, start_patch):
    """Обработка одного региона"""
    print("="*70)
    print(f"📁 Обработка региона: {region_path.name}")
    print("="*70)
    
    files = find_files(region_path)
    
    required = ['before_13ch', 'after_13ch', 'before_3ch', 'after_3ch', 'shp']
    missing = [f for f in required if files[f] is None]
    
    if missing:
        print(f"\n❌ Отсутствуют файлы:")
        for f in missing:
            print(f"   - {f}")
        return 0, start_patch
    
    print(f"\n✅ Все файлы найдены!")
    
    print("\n📷 Загрузка TIFF файлов...")
    
    transform_13ch = None
    transform_3ch = None
    crs = None
    
    try:
        print(f"   📄 {files['after_13ch'].name}")
        with rasterio.open(files['after_13ch']) as src:
            after_13ch = src.read().astype(np.float32)
            transform_13ch = src.transform
            crs = src.crs
            print(f"      Каналы: {after_13ch.shape[0]}, Размер: {after_13ch.shape[2]}×{after_13ch.shape[1]}")
        
        print(f"   📄 {files['after_3ch'].name}")
        with rasterio.open(files['after_3ch']) as src:
            after_3ch = src.read().astype(np.float32)
            transform_3ch = src.transform
            print(f"      Каналы: {after_3ch.shape[0]}, Размер: {after_3ch.shape[2]}×{after_3ch.shape[1]}")
        
        print(f"   📄 {files['before_13ch'].name}")
        with rasterio.open(files['before_13ch']) as src:
            before_13ch = src.read().astype(np.float32)
        
        print(f"   📄 {files['before_3ch'].name}")
        with rasterio.open(files['before_3ch']) as src:
            before_3ch = src.read().astype(np.float32)
        
        print("   ✅ Все TIFF загружены с геопривязкой")
        
    except Exception as e:
        print(f"   ❌ Ошибка при загрузке TIFF: {e}")
        import traceback
        traceback.print_exc()
        return 0, start_patch
    
    mask_path = region_path / "mask_temp.tif"
    try:
        mask = rasterize_shp(files['shp'], files['after_13ch'], mask_path)
    except Exception as e:
        print(f"   ❌ Ошибка при растрировании: {e}")
        import traceback
        traceback.print_exc()
        return 0, start_patch
    
    output_dir_13ch = OUTPUT_DIR / "13_channels"
    output_dir_3ch = OUTPUT_DIR / "3_channels"
    output_dir_13ch.mkdir(parents=True, exist_ok=True)
    output_dir_3ch.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Создание патчей...")
    patch_count, next_patch_num = create_patches(
        after_13ch, after_3ch, mask,
        region_path.name,
        output_dir_13ch, output_dir_3ch,
        before_13ch, before_3ch,
        transform_13ch, transform_3ch, crs,
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
    print(f"   🔥 Мин. процент пожара: {MIN_FIRE_PERCENT}%")
    print(f"   📁 Выход: {OUTPUT_DIR}")
    print("="*70)
    
    if not REGION_PATH.exists():
        print(f"\n❌ Папка региона не найдена: {REGION_PATH}")
        print("💡 Проверь путь в переменной REGION_PATH")
        return
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        print(f"\n📁 Создана папка: {OUTPUT_DIR}")
    
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
        print(f"   3. Запусти скрипт снова")
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
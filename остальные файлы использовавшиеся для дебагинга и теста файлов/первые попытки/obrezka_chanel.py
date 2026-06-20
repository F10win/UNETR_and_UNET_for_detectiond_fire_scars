import tifffile
import numpy as np
from pathlib import Path

# Путь к датасету
DATASET_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\img")

print("="*70)
print("🔧 Исправление файлов after_3ch.tif (4 канала → 3 канала)")
print("="*70)
print(f"\n📂 Путь: {DATASET_PATH}\n")

fixed_count = 0
error_count = 0
skipped_count = 0

# Проходим по всем папкам патчей
for patch_folder in sorted(DATASET_PATH.iterdir()):
    if patch_folder.is_dir():
        after_3ch_path = patch_folder / "S2A_MSIL2A_20230827T020701_N0509_R103_T54VUL_20230827T051653.tif"
        
        if after_3ch_path.exists():
            try:
                # Загружаем файл
                img = tifffile.imread(after_3ch_path)
                
                # Определяем количество каналов
                if img.ndim == 3:
                    # (C, H, W) или (H, W, C)
                    if img.shape[0] < 20:
                        channels = img.shape[0]
                        channels_first = True
                    else:
                        channels = img.shape[2]
                        channels_first = False
                else:
                    channels = 1
                
                if channels == 4:
                    # Удаляем последний канал
                    if channels_first:
                        img_3ch = img[:3, :, :]  # (3, H, W)
                    else:
                        img_3ch = img[:, :, :3]  # (H, W, 3)
                    
                    # Сохраняем обратно
                    tifffile.imwrite(
                        after_3ch_path,
                        img_3ch.astype(np.uint16),
                        compression='lzw',
                        photometric='rgb'
                    )
                    
                    print(f"✅ {patch_folder.name}: 4 канала → 3 канала")
                    fixed_count += 1
                    
                elif channels == 3:
                    print(f"⏭️  {patch_folder.name}: OK (3 канала)")
                    skipped_count += 1
                else:
                    print(f"⚠️  {patch_folder.name}: {channels} каналов (пропущено)")
                    error_count += 1
                    
            except Exception as e:
                print(f"❌ {patch_folder.name}: Ошибка - {e}")
                error_count += 1
        else:
            print(f"❌ {patch_folder.name}: after_3ch.tif не найден")
            error_count += 1

print("\n" + "="*70)
print("📊 ИТОГИ:")
print("="*70)
print(f"✅ Исправлено файлов: {fixed_count}")
print(f"⏭️  Пропущено (уже 3 канала): {skipped_count}")
print(f"❌ Ошибок: {error_count}")
print("="*70)

if fixed_count > 0:
    print(f"\n🎉 {fixed_count} файлов(а) успешно исправлены!")
    print("Теперь можно запускать обучение.")
else:
    print("\n✅ Все файлы уже корректны!")
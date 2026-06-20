from pathlib import Path
import tifffile
import sys

base_path = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\13_channels")
patches = sorted([p for p in base_path.iterdir() if p.is_dir()])

print(f"🔍 Проверка {len(patches)} патчей...")
errors = []

for i, patch in enumerate(patches):
    if i % 500 == 0:
        print(f"Проверено: {i}/{len(patches)}")
    
    # Проверяем оба возможных имени
    img_path = patch / "after_11ch.tif" if (patch / "after_11ch.tif").exists() else patch / "after_13ch.tif"
    mask_path = patch / "mask.tif"
    
    if not img_path.exists():
        errors.append(f"❌ {patch.name}: нет изображения")
        continue
    
    if not mask_path.exists():
        errors.append(f"❌ {patch.name}: нет маски")
        continue
    
    try:
        img = tifffile.imread(img_path)
        if img.size == 0:
            errors.append(f"❌ {patch.name}: пустой TIFF")
    except Exception as e:
        errors.append(f"❌ {patch.name}: {str(e)}")
    
    try:
        mask = tifffile.imread(mask_path)
        if mask.size == 0:
            errors.append(f"❌ {patch.name}: пустая маска")
    except Exception as e:
        errors.append(f"❌ {patch.name} (mask): {str(e)}")

if errors:
    print(f"\n⚠️  Найдено {len(errors)} ошибок:")
    for err in errors[:20]:  # Показываем первые 20
        print(err)
    if len(errors) > 20:
        print(f"... и еще {len(errors) - 20}")
else:
    print("✅ Все файлы читаются корректно!")

sys.exit(0 if not errors else 1)
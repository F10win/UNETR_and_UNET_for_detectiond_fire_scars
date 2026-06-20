import tifffile
import numpy as np
from pathlib import Path

base_path = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels")
print("🔍 Проверка всех TIFF на NaN/Inf...")

bad_files = []

for patch_folder in sorted(base_path.iterdir()):
    if patch_folder.is_dir():
        for file_name in ["after_3ch.tif", "mask.tif"]:
            file_path = patch_folder / file_name
            if file_path.exists():
                try:
                    data = tifffile.imread(file_path)
                    if np.isnan(data).any():
                        print(f"❌ NaN в: {file_path}")
                        bad_files.append(file_path)
                    elif np.isinf(data).any():
                        print(f"❌ Inf в: {file_path}")
                        bad_files.append(file_path)
                except Exception as e:
                    print(f"❌ Ошибка чтения: {file_path} - {e}")

if not bad_files:
    print("✅ Все файлы чисты!")
else:
    print(f"\n⚠️ Найдено {len(bad_files)} проблемных файлов!")
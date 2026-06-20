from pathlib import Path
import tifffile

base_path = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels")
print("🔍 Проверка файлов в 3_channels...\n")

problematic = []

for patch_folder in sorted(base_path.iterdir()):
    if patch_folder.is_dir():
        after_3ch = patch_folder / "after_3ch.tif"
        if after_3ch.exists():
            # 🔥 Убрали 'with' - imread() сразу возвращает numpy array
            img = tifffile.imread(after_3ch)
            
            # Определяем количество каналов
            if img.ndim == 3:
                channels = img.shape[0] if img.shape[0] < 20 else img.shape[-1]
            else:
                channels = 1
                
            if channels != 3:
                print(f"❌ {patch_folder.name}: {channels} каналов (ожидалось 3)")
                problematic.append(patch_folder.name)
        else:
            print(f"❌ {patch_folder.name}: Файл not found")

if not problematic:
    print("\n✅ Все файлы корректны! Проблема была только в синтаксисе диагностики.")
else:
    print(f"\n⚠️ Найдено {len(problematic)} проблемных папок. Их нужно пересоздать или удалить.")
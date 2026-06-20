import os
from pathlib import Path
import re

# ============================================================================
# ⚙️ НАСТРОЙКИ
# ============================================================================
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset")

TRAIN_3CH = BASE_DIR / "validation_3ch"
TRAIN_11CH = BASE_DIR / "validation_11ch"
# ============================================================================

def get_sorted_patches(folder):
    """Возвращает отсортированные по номеру объекты Path"""
    patches = [p for p in folder.iterdir() 
               if p.is_dir() and p.name.startswith("patch_")]
    
    # Сортировка по числу после "patch_"
    def extract_num(p):
        match = re.search(r'patch_(\d+)', p.name)
        return int(match.group(1)) if match else 0
        
    patches.sort(key=extract_num)
    return patches

def rename_patches_sequentially(folder, start_from=1):
    """Безопасное переименование в два этапа, чтобы избежать конфликтов имён"""
    print(f"\n Обработка {folder.name}...")
    patches = get_sorted_patches(folder)
    
    if not patches:
        print("   ⚠️ Папок нет, пропускаем.")
        return 0
        
    # ЭТАП 1: Переименовываем во временные имена (избегаем коллизий)
    temp_prefix = "_tmp_rename_"
    for i, patch in enumerate(patches):
        temp_name = f"{temp_prefix}{i:06d}"
        patch.rename(patch.parent / temp_name)
        
    # ЭТАП 2: Переименовываем в финальные patch_XXXX
    temp_patches = sorted([p for p in folder.iterdir() 
                           if p.name.startswith(temp_prefix)])
                           
    for i, patch in enumerate(temp_patches, start=start_from):
        final_name = f"patch_{i:04d}"
        patch.rename(patch.parent / final_name)
        
    print(f"   ✅ Переименовано {len(temp_patches)} папок. Последний: {final_name}")
    return len(temp_patches)

def main():
    print("="*70)
    print("🏷️  Сквозная нумерация тренировочных патчей")
    print("="*70)
    
    if not TRAIN_3CH.exists() or not TRAIN_11CH.exists():
        print("❌ Папки train не найдены. Запустите сначала Скрипт 1.")
        return
        
    count_3 = len(get_sorted_patches(TRAIN_3CH))
    count_11 = len(get_sorted_patches(TRAIN_11CH))
    
    print(f"📊 Патчей в 3_channels: {count_3}")
    print(f" Патчей в 13_channels: {count_11}")
    
    if count_3 != count_11:
        print("⚠️ Внимание: количество папок в наборах отличается!")
        print("   Это может нарушить синхронизацию данных при обучении.")
        
    confirm = input("\n⚠️ Продолжить переименование? (введите ДА): ")
    if confirm.strip().upper() != "ДА":
        print("❌ Отменено.")
        return
        
    n1 = rename_patches_sequentially(TRAIN_3CH, start_from=1)
    n2 = rename_patches_sequentially(TRAIN_11CH, start_from=1)
    
    print(f"\n{'='*70}")
    print("✅ ГОТОВО!")
    print(f" Итого переименовано: {n1} папок в каждом наборе")
    print(f"🔢 Диапазон: patch_0001 ... patch_{n1:04d}")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
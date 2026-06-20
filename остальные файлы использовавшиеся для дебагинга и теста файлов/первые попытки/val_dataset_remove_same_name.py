import os
from pathlib import Path
import shutil

# ============================================================================
# ⚙️ НАСТРОЙКИ
# ============================================================================
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset")

VAL_3CH = BASE_DIR / "validation_3ch"
VAL_11CH = BASE_DIR / "validation_11ch"

TRAIN_3CH = BASE_DIR / "3_channels"
TRAIN_11CH = BASE_DIR / "13_channels"
# ============================================================================

def get_patch_folders(folder):
    """Возвращает отсортированный список имён папок patch_XXXX"""
    if not folder.exists():
        return []
    return sorted([d.name for d in folder.iterdir() 
                   if d.is_dir() and d.name.startswith("patch_")])

def main():
    print("="*70)
    print("🗑️  Удаление валидационных патчей из тренировочных наборов")
    print("="*70)
    
    # 1. Считываем списки папок из валидации
    val_3ch = get_patch_folders(VAL_3CH)
    val_11ch = get_patch_folders(VAL_11CH)
    
    if not val_3ch and not val_11ch:
        print("❌ В папках validation не найдено патчей. Проверьте пути.")
        return
        
    print(f"📂 validation_3ch: {len(val_3ch)} папок")
    print(f"📂 validation_11ch: {len(val_11ch)} папок")
    
    # 2. Находим общие папки (для безопасности используем пересечение)
    folders_to_remove = sorted(list(set(val_3ch) & set(val_11ch)))
    
    if len(folders_to_remove) == 0:
        print("⚠️ Нет совпадающих папок для удаления.")
        return
        
    print(f"\n📋 Будет удалено {len(folders_to_remove)} папок из train наборов.")
    print(f"   Примеры: {folders_to_remove[:3]} ... {folders_to_remove[-3:]}")
    
    # 3. Подтверждение
    confirm = input("\n⚠️ Это действие необратимо! Удалить? (введите ДА): ")
    if confirm.strip().upper() != "ДА":
        print(" Отменено пользователем.")
        return
        
    # 4. Удаление
    deleted = 0
    for name in folders_to_remove:
        p3 = TRAIN_3CH / name
        p11 = TRAIN_11CH / name
        
        if p3.exists():
            shutil.rmtree(p3)
        if p11.exists():
            shutil.rmtree(p11)
        deleted += 1
        
    print(f"\n✅ Готово! Удалено {deleted} папок из каждого тренировочного набора.")
    print("💡 Теперь запустите Скрипт 2 для переименования оставшихся патчей.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
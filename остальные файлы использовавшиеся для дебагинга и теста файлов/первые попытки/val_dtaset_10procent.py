import os
import shutil
from pathlib import Path
import math

# ============================================================================
# ⚙️ НАСТРОЙКИ
# ============================================================================

# Путь к датасету
DATASET_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset")

# Папки с патчами
TRAIN_11CH_PATH = DATASET_PATH / "13_channels"
TRAIN_3CH_PATH = DATASET_PATH / "3_channels"

# Выходные папки для валидации
VAL_11CH_PATH = DATASET_PATH / "validation_11ch"
VAL_3CH_PATH = DATASET_PATH / "validation_3ch"

# Количество патчей для валидации
NUM_VAL_PATCHES = 737

# ============================================================================

def get_sorted_patches(folder_path):
    """Получить отсортированный список папок с патчами"""
    if not folder_path.exists():
        print(f"❌ Папка не найдена: {folder_path}")
        return []
    
    patches = [p for p in folder_path.iterdir() if p.is_dir() and p.name.startswith("patch_")]
    
    # Сортировка по номеру патча
    patches.sort(key=lambda x: int(x.name.split("_")[1]))
    
    return patches

def select_uniform_patches(patches, num_select):
    """Равномерно выбрать патчи из списка"""
    total = len(patches)
    
    if num_select >= total:
        print(f"⚠️  Запрошено {num_select} патчей, но доступно только {total}")
        print(f"   Будут выбраны все доступные патчи")
        return patches
    
    # Вычисляем шаг для равномерного распределения
    step = total / num_select
    
    selected_indices = []
    for i in range(num_select):
        # Вычисляем индекс с округлением
        idx = int(i * step)
        if idx < total and idx not in selected_indices:
            selected_indices.append(idx)
    
    # Если выбрали меньше чем нужно (из-за округлений), добавляем еще
    if len(selected_indices) < num_select:
        # Добавляем недостающие с конца
        for idx in range(total - 1, -1, -1):
            if idx not in selected_indices and len(selected_indices) < num_select:
                selected_indices.append(idx)
    
    selected_indices.sort()
    selected_patches = [patches[idx] for idx in selected_indices]
    
    return selected_patches

def copy_patch_structure(patch_src_11ch, patch_src_3ch, val_11ch_path, val_3ch_path):
    """Скопировать патч в папки валидации"""
    patch_name = patch_src_11ch.name
    
    # Создаем целевые папки
    val_patch_11ch = val_11ch_path / patch_name
    val_patch_3ch = val_3ch_path / patch_name
    
    val_patch_11ch.mkdir(parents=True, exist_ok=True)
    val_patch_3ch.mkdir(parents=True, exist_ok=True)
    
    # Копируем файлы для 11 каналов
    files_11ch = ["after_11ch.tif", "mask.tif"]
    for file_name in files_11ch:
        src_file = patch_src_11ch / file_name
        if src_file.exists():
            shutil.copy2(src_file, val_patch_11ch / file_name)
    
    # Копируем файлы для 3 каналов
    files_3ch = ["after_3ch.tif", "mask.tif"]
    for file_name in files_3ch:
        src_file = patch_src_3ch / file_name
        if src_file.exists():
            shutil.copy2(src_file, val_patch_3ch / file_name)

def main():
    print("="*70)
    print("📊 Создание валидационной выборки")
    print("="*70)
    print(f"\n⚙️  НАСТРОЙКИ:")
    print(f"   📂 Датасет: {DATASET_PATH}")
    print(f"   📈 Всего патчей для валидации: {NUM_VAL_PATCHES}")
    print(f"   📁 Валидация 11ch: {VAL_11CH_PATH}")
    print(f"   📁 Валидация 3ch: {VAL_3CH_PATH}")
    print("="*70)
    
    # Проверяем существование папок
    if not TRAIN_11CH_PATH.exists() or not TRAIN_3CH_PATH.exists():
        print(f"\n❌ Папки с тренировочными данными не найдены!")
        print(f"   Проверьте пути:")
        print(f"   - {TRAIN_11CH_PATH}")
        print(f"   - {TRAIN_3CH_PATH}")
        return
    
    # Получаем списки патчей
    print(f"\n🔍 Сканирование папок...")
    patches_11ch = get_sorted_patches(TRAIN_11CH_PATH)
    patches_3ch = get_sorted_patches(TRAIN_3CH_PATH)
    
    print(f"   ✅ Найдено патчей (11ch): {len(patches_11ch)}")
    print(f"   ✅ Найдено патчей (3ch): {len(patches_3ch)}")
    
    if len(patches_11ch) != len(patches_3ch):
        print(f"\n⚠️  Предупреждение: разное количество патчей в папках!")
        print(f"   Это может привести к ошибкам")
    
    total_patches = len(patches_11ch)
    print(f"\n📊 Всего патчей в датасете: {total_patches}")
    print(f"🎯 Патчей для валидации: {NUM_VAL_PATCHES}")
    print(f"📈 Процент валидации: {(NUM_VAL_PATCHES/total_patches)*100:.2f}%")
    
    # Выбираем патчи равномерно
    print(f"\n📋 Выбор равномерных патчей...")
    selected_patches_11ch = select_uniform_patches(patches_11ch, NUM_VAL_PATCHES)
    selected_patches_3ch = select_uniform_patches(patches_3ch, NUM_VAL_PATCHES)
    
    print(f"   ✅ Выбрано патчей (11ch): {len(selected_patches_11ch)}")
    print(f"   ✅ Выбрано патчей (3ch): {len(selected_patches_3ch)}")
    
    # Показываем номера выбранных патчей
    selected_numbers = [p.name for p in selected_patches_11ch]
    print(f"\n📋 Номера выбранных патчей:")
    print(f"   Первые 10: {', '.join(selected_numbers[:10])}")
    print(f"   ...")
    print(f"   Последние 10: {', '.join(selected_numbers[-10:])}")
    
    # Создаем папки для валидации
    VAL_11CH_PATH.mkdir(parents=True, exist_ok=True)
    VAL_3CH_PATH.mkdir(parents=True, exist_ok=True)
    
    # Копируем патчи
    print(f"\n📋 Копирование патчей...")
    copied_count = 0
    
    for i, (patch_11ch, patch_3ch) in enumerate(zip(selected_patches_11ch, selected_patches_3ch)):
        try:
            copy_patch_structure(
                patch_11ch, 
                patch_3ch, 
                VAL_11CH_PATH, 
                VAL_3CH_PATH
            )
            copied_count += 1
            
            if (i + 1) % 100 == 0 or (i + 1) == len(selected_patches_11ch):
                print(f"   Прогресс: {i + 1}/{len(selected_patches_11ch)} патчей")
                
        except Exception as e:
            print(f"   ❌ Ошибка при копировании {patch_11ch.name}: {e}")
    
    print(f"\n{'='*70}")
    print("✅ ГОТОВО!")
    print("="*70)
    print(f"📊 Скопировано патчей: {copied_count}")
    print(f"📁 Валидация 11ch: {VAL_11CH_PATH}")
    print(f"📁 Валидация 3ch: {VAL_3CH_PATH}")
    print(f"\n💡 Теперь у вас есть:")
    print(f"   - Тренировочная выборка: {total_patches - copied_count} патчей")
    print(f"   - Валидационная выборка: {copied_count} патчей")
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
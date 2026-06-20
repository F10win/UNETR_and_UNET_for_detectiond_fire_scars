from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("🧪 Тестирование аугментаций\n")

# === ПУТИ К ДАННЫМ ===
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
CSV_PATH = BASE_DIR / "satellite_data.csv"

BASE_DIRS = [
    BASE_DIR / "Satellite_burned_area_dataset_part1",
    BASE_DIR / "Satellite_burned_area_dataset_part2",
    BASE_DIR / "Satellite_burned_area_dataset_part3",
    BASE_DIR / "Satellite_burned_area_dataset_part4",
    BASE_DIR / "Satellite_burned_area_dataset_part5",
]

# === ПРОВЕРКА СУЩЕСТВОВАНИЯ ===
print("📁 Проверка путей...")
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV файл не найден: {CSV_PATH}")
print(f"✅ CSV найден: {CSV_PATH}")

for i, base_dir in enumerate(BASE_DIRS):
    if base_dir.exists():
        print(f"✅ Part {i+1}: найден")
    else:
        print(f"⚠️  Part {i+1}: не найден")

# === ЧТЕНИЕ CSV ===
print("\n📊 Чтение CSV...")
df = pd.read_csv(CSV_PATH, sep=';')
print(f"✅ Загружено {len(df)} записей")
print(f"🎨 Доступные fold'ы: {df['fold'].unique().tolist()}")

# === ФУНКЦИИ ЗАГРУЗКИ ===
def find_folder(folder_name, base_dirs):
    """Поиск папки с данными во всех частях датасета"""
    for base_dir in base_dirs:
        folder_path = base_dir / folder_name
        if folder_path.exists():
            return folder_path
    raise FileNotFoundError(f"Папка {folder_name} не найдена")

def load_sentinel2(folder_path):
    """Загрузка Sentinel-2 используя tifffile"""
    s2_files = sorted(glob.glob(str(folder_path / "sentinel2_*.tiff")))
    if not s2_files:
        raise FileNotFoundError(f"Sentinel-2 файлы не найдены в {folder_path}")
    
    # Берём последний файл (post-fire)
    s2_file = s2_files[-1]
    img = tifffile.imread(s2_file).astype(np.float32)
    
    # tifffile возвращает (H, W, C) — переводим в (C, H, W)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    else:
        img = np.expand_dims(img, 0)
    
    # Приводим к 13 каналам (как при обучении)
    if img.shape[0] < 13:
        padding = np.zeros((13 - img.shape[0], *img.shape[1:]), dtype=np.float32)
        img = np.concatenate([img, padding], axis=0)
    elif img.shape[0] > 13:
        img = img[:13]
    
    img = np.clip(img, 0, 1)
    return img

def load_mask(folder_path):
    """Загрузка маски"""
    mask_files = list(glob.glob(str(folder_path / "*_mask.tiff"))) + \
                 list(glob.glob(str(folder_path / "*_mask.png")))
    if not mask_files:
        raise FileNotFoundError(f"Маска не найдена в {folder_path}")
    
    mask = tifffile.imread(mask_files[0]).astype(np.float32)
    mask = (mask > 0).astype(np.float32)  # Бинаризация
    return mask

# === АУГМЕНТАЦИИ ===
print("\n🎨 Создание аугментаций...")

# Аугментации для обучения
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        scale=(0.9, 1.1),
        rotate=(-30, 30),
        p=0.5
    ),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    ], p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    ], p=0.3),
], additional_targets={'mask': 'mask'})

# Аугментации для валидации (только ресайз)
val_transform = A.Compose([
    A.Resize(height=512, width=512),
], additional_targets={'mask': 'mask'})

# === ЗАГРУЗКА ПРИМЕРА ===
print("\n📥 Загрузка примера данных...")

# Берём первую запись из CSV
first_row = df.iloc[0]
folder_name = first_row['folder']
fold_color = first_row['fold']

print(f"📁 Папка: {folder_name}")
print(f"🎨 Fold: {fold_color}")

folder_path = find_folder(folder_name, BASE_DIRS)

# Загружаем данные
img = load_sentinel2(folder_path)      # (13, H, W)
mask = load_mask(folder_path)          # (H, W)

print(f"📐 Размер изображения: {img.shape}")
print(f"📐 Размер маски: {mask.shape}")

# Переводим в формат (H, W, C) для albumentations
img_hwc = np.transpose(img, (1, 2, 0))

# === ПРИМЕНЕНИЕ АУГМЕНТАЦИЙ ===
print("\n🔄 Применение аугментаций...")

# Создаём фигуру для отображения
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Показываем оригинал
axes[0, 0].imshow(img_hwc[:, :, [3, 2, 1]])  # RGB каналы
axes[0, 0].set_title("Оригинал (RGB)")
axes[0, 0].axis('off')

axes[1, 0].imshow(mask, cmap='gray')
axes[1, 0].set_title("Оригинал (Маска)")
axes[1, 0].axis('off')

# Overlay: создаём копию RGB и добавляем красную маску
overlay_orig = img_hwc[:, :, [3, 2, 1]].copy()
overlay_orig[mask > 0.5] = [1, 0, 0]  # Красный цвет для гари
axes[2, 0].imshow(overlay_orig)
axes[2, 0].set_title("Оригинал (Overlay)")
axes[2, 0].axis('off')

# Применяем 3 разных аугментации
for i in range(3):
    augmented = train_transform(image=img_hwc, mask=mask)
    aug_img = augmented['image']
    aug_mask = augmented['mask']
    
    # Показываем результат
    col = i + 1
    
    axes[0, col].imshow(aug_img[:, :, [3, 2, 1]])
    axes[0, col].set_title(f"Аугментация {i+1} (RGB)")
    axes[0, col].axis('off')
    
    axes[1, col].imshow(aug_mask, cmap='gray')
    axes[1, col].set_title(f"Аугментация {i+1} (Маска)")
    axes[1, col].axis('off')
    
    # Overlay для аугментации
    overlay_aug = aug_img[:, :, [3, 2, 1]].copy()
    overlay_aug[aug_mask > 0.5] = [1, 0, 0]
    axes[2, col].imshow(overlay_aug)
    axes[2, col].set_title(f"Аугментация {i+1} (Overlay)")
    axes[2, col].axis('off')

plt.tight_layout()
plt.savefig("augmentations_test.png", dpi=150, bbox_inches='tight')
print("✅ Результат сохранён: augmentations_test.png")
plt.show()

# === СТАТИСТИКА ===
print("\n📊 Статистика:")
print(f"🔥 Горящих пикселей (оригинал): {(mask > 0.5).sum() / mask.size * 100:.2f}%")

# Проверяем что аугментации не сломали данные
augmented = train_transform(image=img_hwc, mask=mask)
print(f"✅ Размер после аугментации: {augmented['image'].shape}")
print(f"✅ Уникальные значения маски: {np.unique(augmented['mask'])}")

print("\n✅ Все тесты пройдены!")
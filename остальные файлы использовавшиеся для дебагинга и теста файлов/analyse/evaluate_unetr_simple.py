#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка модели UNETR - Простая версия
"""

import torch
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import UNETR
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
import os

print("="*70)
print("🔍 ОЦЕНКА МОДЕЛИ UNETR (3 канала)")
print("="*70)

# ========== ПУТИ ==========
MODEL_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\unetr_3ch_fold1.pth")
DATASET_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels")
RESULTS_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\analysis\results")

print(f"\n📂 Dataset: {DATASET_PATH}")
print(f"💾 Model: {MODEL_PATH}")
print(f"📁 Results: {RESULTS_DIR}")

# Создаем папку
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Папка создана/существует")
except Exception as e:
    print(f"❌ Ошибка создания папки: {e}")
    input("Нажмите Enter...")
    sys.exit(1)

# ========== ПАРАМЕТРЫ ==========
BATCH_SIZE = 1
IN_CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

print(f"\n🔧 Device: {DEVICE}")
print(f"🔢 Channels: {IN_CHANNELS}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print("="*70)

# ========== DATASET ==========
class SimpleDataset(Dataset):
    def __init__(self, base_path):
        self.patches = sorted([p for p in Path(base_path).iterdir() if p.is_dir()])
        print(f"📦 Найдено патчей: {len(self.patches)}")
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_folder = self.patches[idx]
        image = tifffile.imread(patch_folder / "after_3ch.tif").astype(np.float32)
        mask = tifffile.imread(patch_folder / "mask.tif").astype(np.float32)
        
        image = np.clip(image / 4000.0, 0, 1)
        mask = (mask > 0).astype(np.float32)
        
        if image.ndim == 3 and image.shape[0] >= 20:
            image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image), torch.from_numpy(mask), patch_folder.name

# ========== ЗАГРУЗКА МОДЕЛИ ==========
print("\n📥 Загрузка модели...")
try:
    model = UNETR(
        in_channels=IN_CHANNELS,
        out_channels=1,
        img_size=(512, 512),
        spatial_dims=2,
        feature_size=16,
        norm_name="instance"
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    input("Нажмите Enter...")
    sys.exit(1)

# ========== ОЦЕНКА ==========
dataset = SimpleDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n🔄 Оценка {len(dataset)} патчей...\n")

all_results = []

with torch.no_grad():
    for i, (images, masks, names) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
        gt = masks.squeeze(1).cpu().numpy()
        
        for j in range(probs.shape[0]):
            pred_binary = (probs[j] > THRESHOLD).astype(np.uint8).flatten()
            gt_binary = gt[j].astype(np.uint8).flatten()
            
            tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary, labels=[0, 1]).ravel()
            
            eps = 1e-8
            iou = tp / (tp + fp + fn + eps)
            dice = 2*tp / (2*tp + fp + fn + eps)
            
            all_results.append({
                'patch': names[j],
                'IoU': iou,
                'Dice': dice,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
                'TN': int(tn)
            })
        
        if (i + 1) % 500 == 0:
            print(f"  Обработано: {len(all_results)}/{len(dataset)}")

print(f"\n✅ Оценка завершена! Всего: {len(all_results)}")

# ========== СОХРАНЕНИЕ ==========
print("\n💾 Сохранение результатов...")

try:
    # 1. Создаем DataFrame
    print("  1. Создание DataFrame...")
    df = pd.DataFrame(all_results)
    print(f"     ✅ DataFrame создан: {len(df)} строк")
    
    # 2. Добавляем метрики
    print("  2. Добавление метрик...")
    df['Precision'] = df['TP'] / (df['TP'] + df['FP'] + eps)
    df['Recall'] = df['TP'] / (df['TP'] + df['FN'] + eps)
    df['F1'] = 2 * df['Precision'] * df['Recall'] / (df['Precision'] + df['Recall'] + eps)
    df['Accuracy'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'] + eps)
    print("     ✅ Метрики добавлены")
    
    # 3. Сохраняем CSV
    csv_path = RESULTS_DIR / "evaluation_results.csv"
    print(f"  3. Сохранение CSV: {csv_path}")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Проверяем создан ли файл
    if csv_path.exists():
        print(f"     ✅ CSV сохранен (размер: {csv_path.stat().st_size} байт)")
    else:
        print(f"     ❌ CSV НЕ создан!")
    
    # 4. Вывод статистики
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЕ МЕТРИКИ:")
    print("="*70)
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Dice', 'IoU']:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col:12s}: {mean:.4f} ± {std:.4f}")
    
    # 5. График
    print("\n📈 Создание графика...")
    png_path = RESULTS_DIR / "metrics_distribution.png"
    
    plt.figure(figsize=(10, 6))
    metrics = ['IoU', 'Dice', 'F1', 'Precision', 'Recall']
    plt.boxplot([df[m] for m in metrics], labels=metrics, patch_artist=True)
    plt.title('Metrics Distribution')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    if png_path.exists():
        print(f"     ✅ График сохранен: {png_path}")
    else:
        print(f"     ❌ График НЕ создан!")
    
    print("\n" + "="*70)
    print("🎉 ГОТОВО!")
    print("="*70)
    print(f"📁 Папка результатов: {RESULTS_DIR}")
    
except Exception as e:
    print(f"\n❌ ОШИБКА при сохранении: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
input("Нажмите Enter для выхода...")
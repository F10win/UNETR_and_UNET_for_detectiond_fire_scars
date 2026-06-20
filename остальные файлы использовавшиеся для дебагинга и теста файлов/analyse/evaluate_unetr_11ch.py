#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка модели UNETR (11 каналов) по метрикам: IoU, Dice, Precision, Recall, F1-Score
"""

import torch
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from monai.networks.nets import UNETR
import os
import warnings
warnings.filterwarnings('ignore')

# === ⚙️ НАСТРОЙКИ ===
CONFIG = {
    'model_path': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\unetr_11ch_fold3.pth"),
    'dataset_dir': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\13_channels"),
    'output_csv': "evaluation_metrics_11ch.csv",
    'output_plots_dir': "plots_11ch",
    'batch_size': 1,
    'threshold': 0.5,
    'num_workers': 2
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Устройство: {DEVICE} | Каналов: 11")

# === 📦 DATASET ===
class BurnEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.patches = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        print(f"📂 Найдено тестовых патчей: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_folder = self.patches[idx]
        
        # Индексы каналов (если нужно выбрать из 13)
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        
        full_image = tifffile.imread(patch_folder / "after_13ch.tif").astype(np.float32)
        mask = tifffile.imread(patch_folder / "mask.tif").astype(np.float32)
        
        # === ИСПРАВЛЕННАЯ ЛОГИКА ===
        if full_image.ndim == 2:
            # Если grayscale, добавляем канал
            image = np.expand_dims(full_image, axis=-1)
        
        if full_image.shape[-1] == 11:
            # Уже 11 каналов в формате (H, W, 11)
            image = full_image
            # Транспонируем в (11, H, W)
            image = np.transpose(image, (2, 0, 1))
        elif full_image.shape[0] == 11:
            # Уже 11 каналов в формате (11, H, W)
            image = full_image
        elif full_image.shape[-1] == 13:
            # 13 каналов в формате (H, W, 13) -> выбираем 11
            image = full_image[:, :, indices]
            image = np.transpose(image, (2, 0, 1))
        elif full_image.shape[0] == 13:
            # 13 каналов в формате (13, H, W) -> выбираем 11
            image = full_image[indices, :, :]
        else:
            raise ValueError(f"Не удалось определить формат изображения. Форма: {full_image.shape}")
        
        # Проверка
        if image.shape[0] != 11:
            raise ValueError(f"Ожидалось 11 каналов, получено: {image.shape[0]}")
        # ========================

        # Нормализация
        image = np.clip(image / 4000.0, 0, 1)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask), patch_folder.name

# === 📊 МЕТРИКИ ===
def compute_metrics(pred_prob, gt_mask, threshold=0.5):
    pred_bin = (pred_prob > threshold).astype(np.uint8).flatten()
    gt_bin = gt_mask.astype(np.uint8).flatten()
    
    tn, fp, fn, tp = confusion_matrix(gt_bin, pred_bin, labels=[0, 1]).ravel()
    eps = 1e-8
    
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return {
        'IoU': iou, 'Dice': dice, 'Precision': precision, 
        'Recall': recall, 'F1_Score': f1,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
    }

# === 📈 ВИЗУАЛИЗАЦИЯ ===
def create_plots(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['IoU'], bins=50, kde=True, color='skyblue')
    plt.title('Распределение IoU по патчам (11 каналов)', fontsize=14)
    plt.xlabel('IoU Score', fontsize=12)
    plt.ylabel('Количество патчей', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'iou_distribution.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']
    sns.boxplot(data=df[metrics_to_plot], palette='Set2')
    plt.title('Сравнение метрик качества (11 каналов)', fontsize=14)
    plt.ylabel('Значение метрики', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    
    print(f"✅ Графики сохранены в папку: {save_dir}")

# === 🔍 ОЦЕНКА ===
def evaluate():
    print("="*70)
    print("🔍 Оценка модели UNETR (11 каналов)")
    print("="*70)

    print("\n📥 Загрузка модели...")
    model = UNETR(
        in_channels=11, out_channels=1, img_size=(512, 512),
        spatial_dims=2, feature_size=16, norm_name="instance"
    ).to(DEVICE)
    
    if not CONFIG['model_path'].exists():
        raise FileNotFoundError(f"❌ Модель не найдена: {CONFIG['model_path']}")
        
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=DEVICE, weights_only=True)
    except Exception:
        checkpoint = torch.load(CONFIG['model_path'], map_location=DEVICE, weights_only=False)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена")

    dataset = BurnEvalDataset(CONFIG['dataset_dir'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                                         num_workers=CONFIG['num_workers'], pin_memory=True)

    print(f"\n🔄 Запуск оценки на {len(dataset)} патчах...")
    all_metrics = []

    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for i in range(len(names)):
                metrics = compute_metrics(probs[i], masks_np[i], CONFIG['threshold'])
                metrics['patch_name'] = names[i]
                all_metrics.append(metrics)

    print("\n📊 Обработка результатов...")
    df = pd.DataFrame(all_metrics)
    
    summary = df[['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']].agg(['mean', 'std', 'min', 'max']).T
    summary.rename(columns={'mean': 'Среднее', 'std': 'Стд.откл.', 'min': 'Мин', 'max': 'Макс'}, inplace=True)
    
    print("\n" + "="*70)
    print("📈 ИТОГОВЫЕ МЕТРИКИ (11 КАНАЛОВ)")
    print("="*70)
    print(summary.round(4))
    print("="*70)
    
    df.to_csv(CONFIG['output_csv'], index=False, encoding='utf-8-sig')
    print(f"\n💾 Детальные результаты сохранены: {CONFIG['output_csv']}")
    
    create_plots(df, CONFIG['output_plots_dir'])
    
    print("✅ Оценка завершена!")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
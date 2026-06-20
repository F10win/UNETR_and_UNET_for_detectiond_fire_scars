#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка модели UNETR (3 канала) по метрикам: IoU, Dice, Precision, Recall, F1-Score
Оптимизирован для GPU и быстрой работы
"""

import torch
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNETR
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# === ⚙️ НАСТРОЙКИ ===
CONFIG = {
    'model_path': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\unetr_3ch_fold1.pth"),
    'test_dir': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels"),
    'batch_size': 4,          # Увеличь если хватает VRAM
    'threshold': 0.5,         # Порог бинаризации
    'num_workers': 2,         # Потоки загрузки данных
    'save_csv': "evaluation_metrics_3ch.csv"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Устройство: {DEVICE} | Batch Size: {CONFIG['batch_size']}")

# === 📦 DATASET ===
class BurnEvalDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.patches = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        print(f"📂 Найдено тестовых патчей: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_folder = self.patches[idx]
        
        # Загрузка
        image = tifffile.imread(patch_folder / "after_3ch.tif").astype(np.float32)
        mask = tifffile.imread(patch_folder / "mask.tif").astype(np.float32)
        
        # Нормализация
        image = np.clip(image / 4000.0, 0, 1)
        mask = (mask > 0).astype(np.float32)
        
        # Приведение к формату (C, H, W)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
            
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)
        
        return torch.from_numpy(image), torch.from_numpy(mask), patch_folder.name

# === 📊 МЕТРИКИ ===
def compute_metrics(pred_prob, gt_mask, threshold=0.5):
    """Вычисляет бинарные метрики для одного патча"""
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
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
    }

# === 🔍 ОЦЕНКА ===
def evaluate():
    print("="*70)
    print("🔍 Оценка модели UNETR (3 канала)")
    print("="*70)

    # 1. Загрузка модели
    print("\n📥 Загрузка модели...")
    model = UNETR(
        in_channels=3, out_channels=1, img_size=(512, 512),
        spatial_dims=2, feature_size=16, norm_name="instance"
    ).to(DEVICE)
    
    if not CONFIG['model_path'].exists():
        raise FileNotFoundError(f"❌ Модель не найдена: {CONFIG['model_path']}")
        
    checkpoint = torch.load(CONFIG['model_path'], map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена")

    # 2. Dataset & DataLoader
    dataset = BurnEvalDataset(CONFIG['test_dir'])
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True)

    # 3. Инференс
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

    # 4. Агрегация результатов
    print("\n📊 Обработка результатов...")
    df = pd.DataFrame(all_metrics)
    
    # Сводная таблица
    summary = df[['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']].agg(['mean', 'std', 'min', 'max']).T
    summary.rename(columns={'mean': 'Среднее', 'std': 'Стд.откл.', 'min': 'Мин', 'max': 'Макс'}, inplace=True)
    
    print("\n" + "="*70)
    print("📈 ИТОГОВЫЕ МЕТРИКИ")
    print("="*70)
    print(summary.round(4))
    print("="*70)
    
    # 5. Сохранение
    csv_path = Path(CONFIG['save_csv'])
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Детальные результаты сохранены: {csv_path}")
    print(f"📁 Всего обработано патчей: {len(df)}")
    print("✅ Оценка завершена!")

if __name__ == "__main__":
    try:
        evaluate()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
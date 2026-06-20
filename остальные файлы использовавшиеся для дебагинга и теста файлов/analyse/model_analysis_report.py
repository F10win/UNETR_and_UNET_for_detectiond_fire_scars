#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ и визуализация результатов оценки модели UNETR
"""

import torch
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import UNETR
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("🔍 АНАЛИЗ И ВИЗУАЛИЗАЦИЯ МОДЕЛИ UNETR")
print("="*80)

# === ПУТИ ===
MODEL_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\unetr_3ch_fold1.pth")
DATASET_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels")
REPORT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\analysis\model_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# === ПАРАМЕТРЫ ===
BATCH_SIZE = 1
IN_CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

print(f"\n📂 Dataset: {DATASET_PATH}")
print(f"💾 Model: {MODEL_PATH}")
print(f"📁 Report: {REPORT_DIR}")
print(f"🔧 Device: {DEVICE}")

# === DATASET ===
class PatchDataset(Dataset):
    def __init__(self, base_path):
        self.patches = sorted([p for p in Path(base_path).iterdir() if p.is_dir()])
        
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

# === ЗАГРУЗКА МОДЕЛИ ===
print("\n📥 Загрузка модели...")
model = UNETR(
    in_channels=IN_CHANNELS, out_channels=1, img_size=(512, 512),
    spatial_dims=2, feature_size=16, norm_name="instance"
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Извлечение информации о модели
model_info = {
    'epoch': checkpoint.get('epoch', 'N/A'),
    'val_loss': checkpoint.get('val_loss', 'N/A'),
    'total_params': sum(p.numel() for p in model.parameters()),
    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024)
}

print(f"✅ Модель загружена")
print(f"   Эпоха: {model_info['epoch']}")
print(f"   Val Loss: {model_info['val_loss']:.4f}")
print(f"   Параметры: {model_info['total_params']:,}")

# === ОЦЕНКА ===
dataset = PatchDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n🔄 Оценка {len(dataset)} патчей...")

all_results = []
all_preds = []
all_gts = []

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
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2*precision*recall / (precision + recall + eps)
            accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
            
            all_results.append({
                'patch': names[j],
                'IoU': iou,
                'Dice': dice,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Accuracy': accuracy,
                'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
            })
            all_preds.append(probs[j])
            all_gts.append(gt[j])

df = pd.DataFrame(all_results)
print(f"\n✅ Оценка завершена!")

# === ГРАФИК 1: Распределение метрик (Boxplot) ===
print("\n📊 Создание графиков...")

plt.figure(figsize=(14, 8))
metrics = ['IoU', 'Dice', 'F1', 'Precision', 'Recall', 'Accuracy']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

boxplot = plt.boxplot([df[m] for m in metrics], labels=metrics, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('Значение метрики', fontsize=14, fontweight='bold')
plt.title('📊 Распределение метрик модели UNETR', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y')

# Добавление средних значений
for i, m in enumerate(metrics):
    mean_val = df[m].mean()
    plt.text(i+1, mean_val + 0.02, f'{mean_val:.3f}', ha='center', 
             fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig(REPORT_DIR / "metrics_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()

# === ГРАФИК 2: Гистограмма распределения IoU ===
plt.figure(figsize=(12, 6))
plt.hist(df['IoU'], bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7, density=False)
plt.axvline(df['IoU'].mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {df["IoU"].mean():.3f}')
plt.axvline(df['IoU'].median(), color='blue', linestyle=':', linewidth=2, label=f'Медиана: {df["IoU"].median():.3f}')
plt.xlabel('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
plt.ylabel('Количество патчей', fontsize=14, fontweight='bold')
plt.title('📈 Распределение IoU по патчам', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(REPORT_DIR / "iou_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# === ГРАФИК 3: Корреляция метрик ===
plt.figure(figsize=(10, 8))
corr = df[['IoU', 'Dice', 'Precision', 'Recall', 'F1']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.3f', 
            linewidths=0.5, square=True, vmin=-1, vmax=1,
            annot_kws={'size': 12, 'weight': 'bold'})
plt.title('🔗 Корреляция между метриками', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(REPORT_DIR / "metrics_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# === ГРАФИК 4: Scatter plot IoU vs Precision/Recall ===
plt.figure(figsize=(12, 6))
plt.scatter(df['Precision'], df['Recall'], c=df['IoU'], cmap='viridis', 
           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.colorbar(label='IoU')
plt.xlabel('Precision', fontsize=14, fontweight='bold')
plt.ylabel('Recall', fontsize=14, fontweight='bold')
plt.title('🎯 Precision vs Recall (цвет = IoU)', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "precision_recall_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

# === ГРАФИК 5: Confusion Matrix (агрегированная) ===
total_tp = df['TP'].sum()
total_fp = df['FP'].sum()
total_fn = df['FN'].sum()
total_tn = df['TN'].sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Абсолютные значения
cm_abs = np.array([[total_tn, total_fp], [total_fn, total_tp]])
sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Background', 'Fire'], yticklabels=['Background', 'Fire'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Абсолютные)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Проценты
cm_percent = cm_abs.astype('float') / cm_abs.sum() * 100
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Reds', ax=axes[1],
            xticklabels=['Background', 'Fire'], yticklabels=['Background', 'Fire'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.suptitle('📊 Confusion Matrix', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# === ГРАФИК 6: Информация о модели ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

info_text = f"""
🤖 МОДЕЛЬ UNETR - ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ

📌 Архитектура: UNETR (Vision Transformer + U-Net)
📌 Входные каналы: {IN_CHANNELS} (SWIR, NIR, Red)
📌 Размер патча: 512×512 пикселей
📌 Loss функция: Dice Loss
📌 Optimizer: AdamW
📌 Learning Rate: 5e-5

📊 ПАРАМЕТРЫ МОДЕЛИ:
   • Всего параметров: {model_info['total_params']:,}
   • Обучаемых параметров: {model_info['trainable_params']:,}
   • Размер весов: {model_info['model_size_mb']:.2f} MB

🎯 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:
   • Эпоха сохранения: {model_info['epoch']}
   • Val Loss: {model_info['val_loss']:.4f}
   • Dice Score ≈ {1 - model_info['val_loss']:.4f}

📈 ИТОГОВЫЕ МЕТРИКИ (на {len(df)} патчах):
   • IoU (среднее): {df['IoU'].mean():.4f} ± {df['IoU'].std():.4f}
   • Dice (среднее): {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}
   • F1-Score: {df['F1'].mean():.4f} ± {df['F1'].std():.4f}
   • Precision: {df['Precision'].mean():.4f} ± {df['Precision'].std():.4f}
   • Recall: {df['Recall'].mean():.4f} ± {df['Recall'].std():.4f}
   • Accuracy: {df['Accuracy'].mean():.4f} ± {df['Accuracy'].std():.4f}
"""

ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.title('📋 Паспорт модели UNETR', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(REPORT_DIR / "model_info.png", dpi=300, bbox_inches='tight')
plt.close()

# === СОХРАНЕНИЕ CSV ===
df.to_csv(REPORT_DIR / "evaluation_detailed.csv", index=False)

# === СОЗДАНИЕ ОТЧЕТА (HTML) ===
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Отчет: Модель UNETR для сегментации гарей</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-card {{ display: inline-block; background: #ecf0f1; padding: 15px 25px; margin: 10px; border-radius: 8px; text-align: center; min-width: 150px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        img {{ max-width: 100%; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Отчет: Модель UNETR для сегментации лесных гарей</h1>
        <p><strong>Дата:</strong> {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}</p>
        <p><strong>Датасет:</strong> {len(df)} патчей (512×512, 3 канала)</p>
        
        <h2>📊 Ключевые метрики</h2>
        <div class="metric-card">
            <div class="metric-value">{df['IoU'].mean():.3f}</div>
            <div class="metric-label">IoU (среднее)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['Dice'].mean():.3f}</div>
            <div class="metric-label">Dice Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['F1'].mean():.3f}</div>
            <div class="metric-label">F1-Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['Precision'].mean():.3f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['Recall'].mean():.3f}</div>
            <div class="metric-label">Recall</div>
        </div>
        
        <h2>📈 Графики</h2>
        <img src="metrics_boxplot.png" alt="Распределение метрик">
        <img src="iou_distribution.png" alt="Распределение IoU">
        <img src="metrics_correlation.png" alt="Корреляция метрик">
        <img src="precision_recall_scatter.png" alt="Precision vs Recall">
        <img src="confusion_matrix.png" alt="Confusion Matrix">
        <img src="model_info.png" alt="Информация о модели">
        
        <h2>📋 Детальная статистика</h2>
        <table>
            <tr><th>Метрика</th><th>Среднее</th><th>Медиана</th><th>Std</th><th>Мин</th><th>Макс</th></tr>
            <tr><td>IoU</td><td>{df['IoU'].mean():.4f}</td><td>{df['IoU'].median():.4f}</td><td>{df['IoU'].std():.4f}</td><td>{df['IoU'].min():.4f}</td><td>{df['IoU'].max():.4f}</td></tr>
            <tr><td>Dice</td><td>{df['Dice'].mean():.4f}</td><td>{df['Dice'].median():.4f}</td><td>{df['Dice'].std():.4f}</td><td>{df['Dice'].min():.4f}</td><td>{df['Dice'].max():.4f}</td></tr>
            <tr><td>F1</td><td>{df['F1'].mean():.4f}</td><td>{df['F1'].median():.4f}</td><td>{df['F1'].std():.4f}</td><td>{df['F1'].min():.4f}</td><td>{df['F1'].max():.4f}</td></tr>
            <tr><td>Precision</td><td>{df['Precision'].mean():.4f}</td><td>{df['Precision'].median():.4f}</td><td>{df['Precision'].std():.4f}</td><td>{df['Precision'].min():.4f}</td><td>{df['Precision'].max():.4f}</td></tr>
            <tr><td>Recall</td><td>{df['Recall'].mean():.4f}</td><td>{df['Recall'].median():.4f}</td><td>{df['Recall'].std():.4f}</td><td>{df['Recall'].min():.4f}</td><td>{df['Recall'].max():.4f}</td></tr>
        </table>
        
        <h2>🔍 Оценка качества</h2>
        <p><strong>IoU (Intersection over Union):</strong> 
            <span class="{'good' if df['IoU'].mean() > 0.7 else 'warning' if df['IoU'].mean() > 0.5 else 'critical'}">{df['IoU'].mean():.3f}</span>
            {"✅ Отличный результат!" if df['IoU'].mean() > 0.7 else "⚠️ Приемлемый результат" if df['IoU'].mean() > 0.5 else "❌ Требуется улучшение"}
        </p>
        <p><strong>Precision (точность):</strong> {df['Precision'].mean():.3f} - доля верно предсказанных пикселей гари среди всех предсказанных</p>
        <p><strong>Recall (полнота):</strong> {df['Recall'].mean():.3f} - доля найденных пикселей гари среди всех реальных</p>
        <p><strong>F1-Score:</strong> {df['F1'].mean():.3f} - гармоническое среднее Precision и Recall</p>
    </div>
</body>
</html>
"""

with open(REPORT_DIR / "model_report.html", 'w', encoding='utf-8') as f:
    f.write(html_report)

print("\n" + "="*80)
print("🎉 ОТЧЕТ ГОТОВ!")
print("="*80)
print(f"📁 Все файлы сохранены в: {REPORT_DIR}")
print("   📊 metrics_boxplot.png - Распределение метрик")
print("   📈 iou_distribution.png - Гистограмма IoU")
print("   🔗 metrics_correlation.png - Корреляция метрик")
print("   🎯 precision_recall_scatter.png - Precision vs Recall")
print("   📊 confusion_matrix.png - Confusion Matrix")
print("   📋 model_info.png - Паспорт модели")
print("   📄 evaluation_detailed.csv - Детальные результаты")
print("   🌐 model_report.html - HTML отчет")
print("="*80)
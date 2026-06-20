#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка модели UNETR (11 каналов) по метрикам: IoU, Dice, Precision, Recall, F1-Score
Генерация CSV отчета, графиков и текстового отчета на русском языке
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
    # Путь к лучшей модели (11 каналов)
    'model_path': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\unetr_11ch_fold3.pth"),
    # Путь к папке с 13-канальными патчами (скрипт сам найдет after_13ch.tif внутри)
    'dataset_dir': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\13_channels"),
    'output_csv': "evaluation_metrics_11ch.csv",
    'output_plots_dir': "plots_11ch",
    'output_report': "report_11ch.txt",
    'batch_size': 2,
    'threshold': 0.5,
    'num_workers': 2
}

# Индексы каналов для извлечения 11 из 13 (пропускаем B1(0) и B10(10))
# Порядок: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
CHANNEL_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

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

    def __getitem__(
        self, idx):
        patch_folder = self.patches[idx]
        
        # 1. Загрузка файла (называется 13ch, но внутри 11 каналов)
        full_image = tifffile.imread(patch_folder / "after_13ch.tif").astype(np.float32)
        mask = tifffile.imread(patch_folder / "mask.tif").astype(np.float32)
        
        # 2. Нормализация
        image = np.clip(full_image / 4000.0, 0, 1)
        mask = (mask > 0).astype(np.float32)
        
        # 3. ИСПРАВЛЕНИЕ: Обработка размерности
        # tifffile загрузил как (H, W, C) -> (512, 512, 11)
        if image.shape[-1] == 11:
            # Просто переворачиваем оси в (C, H, W) -> (11, 512, 512)
            # Индексы не трогаем, так как каналы уже выбраны
            image = np.transpose(image, (2, 0, 1))
            
        # Если вдруг попадется настоящий 13-канальный файл (на всякий случай)
        elif image.shape[-1] == 13:
            # indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12] # Без B1 и B10
            # image = image[:, :, indices]
            # image = np.transpose(image, (2, 0, 1))
            raise ValueError("Обнаружен файл с 13 каналами. Раскомментируйте логику выбора индексов.")
            
        # Если уже загружено как (C, H, W)
        elif image.shape[0] == 11:
            pass 
        else:
            raise ValueError(f"Неверная размерность изображения: {image.shape}")

        # 4. Подготовка маски (добавляем размерность канала: 1, H, W)
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
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # Для корректного отображения кириллицы
    
    # 1. Распределение IoU
    plt.figure(figsize=(10, 6))
    sns.histplot(df['IoU'], bins=50, kde=True, color='skyblue')
    plt.title('Распределение IoU по патчам (11 каналов)', fontsize=14)
    plt.xlabel('Значение IoU', fontsize=12)
    plt.ylabel('Количество патчей', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'iou_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Boxplot метрик
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']
    sns.boxplot(data=df[metrics_to_plot], palette='Set2')
    plt.title('Сравнение метрик качества (11 каналов)', fontsize=14)
    plt.ylabel('Значение метрики', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    
    print(f"✅ Графики сохранены в папку: {save_dir}")

# === 📄 ГЕНЕРАЦИЯ ОТЧЕТА ===
def generate_report(df, save_path):
    mean_metrics = df[['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']].mean()
    std_metrics = df[['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']].std()
    total_patches = len(df)
    
    # Оценка качества
    iou_mean = mean_metrics['IoU']
    if iou_mean >= 0.80: quality = "Отличное"
    elif iou_mean >= 0.70: quality = "Хорошее"
    elif iou_mean >= 0.50: quality = "Удовлетворительное"
    else: quality = "Требует доработки"
    
    report = f"""
ОТЧЕТ ПО ОЦЕНКЕ МОДЕЛИ UNETR (11 СПЕКТРАЛЬНЫХ КАНАЛОВ)
======================================================
Дата генерации: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
Количество протестированных патчей: {total_patches}
Порог бинаризации: {CONFIG['threshold']}

📊 СРЕДНИЕ ЗНАЧЕНИЯ МЕТРИК:
------------------------------------------------------
{'Метрика':<12} | {'Среднее':<8} | {'Стд. откл.':<10}
------------------------------------------------------
{'IoU':<12} | {mean_metrics['IoU']:.4f}   | {std_metrics['IoU']:.4f}
{'Dice':<12} | {mean_metrics['Dice']:.4f}   | {std_metrics['Dice']:.4f}
{'Precision':<12} | {mean_metrics['Precision']:.4f}   | {std_metrics['Precision']:.4f}
{'Recall':<12} | {mean_metrics['Recall']:.4f}   | {std_metrics['Recall']:.4f}
{'F1-Score':<12} | {mean_metrics['F1_Score']:.4f}   | {std_metrics['F1_Score']:.4f}

🎯 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:
• Качество сегментации: {quality}
• IoU ({iou_mean:.3f}) показывает степень перекрытия предсказанных и реальных контуров гари.
• Precision ({mean_metrics['Precision']:.3f}) характеризует долю ложных срабатываний (чем выше, тем меньше шума).
• Recall ({mean_metrics['Recall']:.3f}) показывает полноту обнаружения реальных пожаров.
• F1-Score ({mean_metrics['F1_Score']:.3f}) отражает баланс между точностью и полнотой.

📈 СТАТИСТИКА РАСПРЕДЕЛЕНИЯ:
• Минимальный IoU: {df['IoU'].min():.4f}
• Максимальный IoU: {df['IoU'].max():.4f}
• Медиана IoU: {df['IoU'].median():.4f}

💡 РЕКОМЕНДАЦИИ:
{'- Модель стабильна и готова к практическому применению.' if iou_mean > 0.75 else '- Рекомендуется дообучение на сложных примерах или корректировка порога.'}
- Для снижения False Positive можно увеличить порог бинаризации до 0.55-0.60.
- Для повышения Recall (полноты) можно снизить порог до 0.40-0.45.
======================================================
"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📄 Текстовый отчет сохранен: {save_path}")

# === 🔍 ОЦЕНКА ===
def evaluate():
    print("="*70)
    print("🔍 Оценка модели UNETR (11 каналов)")
    print("="*70)

    # 1. Загрузка модели
    print("\n📥 Загрузка модели...")
    model = UNETR(
        in_channels=11, out_channels=1, img_size=(512, 512),
        spatial_dims=2, feature_size=16, norm_name="instance"
    ).to(DEVICE)
    
    if not CONFIG['model_path'].exists():
        raise FileNotFoundError(f"❌ Модель не найдена: {CONFIG['model_path']}")
        
    checkpoint = torch.load(CONFIG['model_path'], map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена")

    # 2. Dataset
    dataset = BurnEvalDataset(CONFIG['dataset_dir'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
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

    # 4. Результаты
    print("\n📊 Обработка результатов...")
    df = pd.DataFrame(all_metrics)
    
    summary = df[['IoU', 'Dice', 'Precision', 'Recall', 'F1_Score']].agg(['mean', 'std', 'min', 'max']).T
    summary.rename(columns={'mean': 'Среднее', 'std': 'Стд.откл.', 'min': 'Мин', 'max': 'Макс'}, inplace=True)
    
    print("\n" + "="*70)
    print("📈 ИТОГОВЫЕ МЕТРИКИ (11 КАНАЛОВ)")
    print("="*70)
    print(summary.round(4))
    print("="*70)
    
    # 5. Сохранение CSV
    df.to_csv(CONFIG['output_csv'], index=False, encoding='utf-8-sig')
    print(f"\n💾 Детальные результаты сохранены: {CONFIG['output_csv']}")
    
    # 6. Графики
    create_plots(df, CONFIG['output_plots_dir'])
    
    # 7. Отчет
    generate_report(df, CONFIG['output_report'])
    
    print("\n✅ Оценка завершена!")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
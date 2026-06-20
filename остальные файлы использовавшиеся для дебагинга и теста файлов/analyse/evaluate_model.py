import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import UNETR
from dataset import get_dataloaders
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# === НАСТРОЙКИ ===
MODEL_PATH = "checkpoints/unet_best.pth"
CSV_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\satellite_data.csv")
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
BASE_DIRS = [BASE_DIR / f"Satellite_burned_area_dataset_part{i}" for i in range(1, 6)]

IMG_SIZE = (512, 512)
BATCH_SIZE = 2
IN_CHANNELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_path, device):
    """Загрузка модели"""
    model = UNETR(
        in_channels=IN_CHANNELS,
        out_channels=1,
        img_size=IMG_SIZE,
        spatial_dims=2,
        feature_size=16,
        norm_name="instance"
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    """Вычисление метрик для одного примера"""
    pred_binary = (pred_mask > threshold).astype(np.uint8).flatten()
    gt_binary = gt_mask.astype(np.uint8).flatten()
    
    # Confusion matrix элементы
    tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary, labels=[0, 1]).ravel()
    
    # Метрики
    iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0
    dice = 2*tp / (2*tp + fp + fn + 1e-8) if (2*tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'IoU': iou, 'Dice': dice, 'Precision': precision,
        'Recall': recall, 'F1': f1, 'Accuracy': accuracy,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    }

def evaluate(model, val_loader, device, threshold=0.5):
    """Оценка модели на валидационном наборе"""
    print("🔄 Оценка модели...\n")
    
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            
            # Предсказание
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            gt = masks.squeeze(1).cpu().numpy()
            
            # Метрики для каждого изображения в batch'е
            for i in range(probs.shape[0]):
                metrics = compute_metrics(probs[i], gt[i], threshold)
                all_metrics.append(metrics)
    
    # Агрегация результатов
    df = pd.DataFrame(all_metrics)
    
    print("\n📊 СВОДНЫЕ МЕТРИКИ (среднее ± стандартное отклонение):")
    print("=" * 60)
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Dice', 'IoU']:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col:10s}: {mean:.4f} ± {std:.4f}")
    
    print("\n📋 Детальная статистика:")
    print(df.describe()[['IoU', 'Dice', 'F1']].round(4))
    
    # Сохранение результатов
    df.to_csv("analysis/evaluation_results.csv", index=False)
    print(f"\n✅ Результаты сохранены: analysis/evaluation_results.csv")
    
    return df

def plot_metrics_distribution(df, save_path="analysis/metrics_distribution.png"):
    """График распределения метрик"""
    import matplotlib.pyplot as plt
    
    metrics = ['IoU', 'Dice', 'F1', 'Precision', 'Recall']
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([df[m] for m in metrics], labels=metrics, patch_artist=True)
    plt.ylabel('Значение метрики', fontsize=12)
    plt.title('Распределение метрик на валидационном наборе', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ График распределения сохранён: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("🎯 Оценка модели UNETR\n")
    
    # Загрузка модели
    print("📦 Загрузка модели...")
    model = get_model(MODEL_PATH, DEVICE)
    print(f"✅ Модель загружена на {DEVICE}\n")
    
    # Создание валидационного dataloader'а
    # Используем тот же fold, на котором модель валидировалась
    VAL_FOLDS = ['magenta']  # <-- Поменяй на нужный!
    
    print(f"📂 Валидация на fold'ах: {VAL_FOLDS}")
    
    # Создаём только val_loader (train_folds должны быть непустыми)
    # Поэтому передаём все fold'ы кроме val
    all_folds = ['purple', 'coral', 'pink', 'grey', 'cyan', 'lime', 'magenta']
    train_folds_for_loader = [f for f in all_folds if f not in VAL_FOLDS]

    _, val_loader = get_dataloaders(
        csv_path=CSV_PATH,
        base_dirs=BASE_DIRS,
        train_folds=train_folds_for_loader,  # <-- Исправлено
        val_folds=VAL_FOLDS,
        batch_size=BATCH_SIZE,
        num_workers=2,
        img_size=IMG_SIZE
    )
    
    # Оценка
    results_df = evaluate(model, val_loader, DEVICE)
    
    # Визуализация
    plot_metrics_distribution(results_df)
    
    print("\n🎉 Оценка завершена!")
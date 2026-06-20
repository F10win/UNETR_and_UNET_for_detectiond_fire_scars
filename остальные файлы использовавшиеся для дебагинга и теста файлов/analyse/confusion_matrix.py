import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from monai.networks.nets import UNETR
from dataset import get_dataloaders

# === НАСТРОЙКИ ===
MODEL_PATH = "checkpoints/unetr_best_improved.pth"
CSV_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\satellite_data.csv")
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
BASE_DIRS = [BASE_DIR / f"Satellite_burned_area_dataset_part{i}" for i in range(1, 6)]

IMG_SIZE = (512, 512)
BATCH_SIZE = 2
IN_CHANNELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_path, device):
    model = UNETR(
        in_channels=IN_CHANNELS, out_channels=1, img_size=IMG_SIZE,
        spatial_dims=2, feature_size=16, norm_name="instance"
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def plot_confusion_matrix(cm, class_names, save_path="analysis/confusion_matrix.png"):
    """Построение и сохранение матрицы ошибок"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.title('Матрица ошибок (пиксели)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Матрица ошибок сохранена: {save_path}")
    plt.show()

def evaluate_confusion(model, val_loader, device, threshold=0.5):
    """Сбор статистики для матрицы ошибок"""
    print("🔄 Сбор статистики...\n")
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            gt = masks.squeeze(1).cpu().numpy()
            
            pred_binary = (probs > threshold).astype(np.uint8)
            gt_binary = gt.astype(np.uint8)
            
            all_preds.extend(pred_binary.flatten())
            all_gts.extend(gt_binary.flatten())
    
    # Матрица ошибок
    cm = confusion_matrix(all_gts, all_preds, labels=[0, 1])
    
    # Отчёт
    print("📋 Classification Report (по пикселям):")
    print("-" * 50)
    report = classification_report(all_gts, all_preds, 
                                   target_names=['Не гарь (0)', 'Гарь (1)'],
                                   output_dict=True)
    print(classification_report(all_gts, all_preds, 
                              target_names=['Не гарь (0)', 'Гарь (1)']))
    
    # Визуализация
    plot_confusion_matrix(cm, ['Не гарь', 'Гарь'])
    
    # Статистика дисбаланса
    total = cm.sum()
    burned_gt = np.sum(all_gts)
    burned_pred = np.sum(all_preds)
    
    print(f"\n📊 Статистика:")
    print(f"  Всего пикселей: {total:,}")
    print(f"  Гарь (Ground Truth): {burned_gt:,} ({burned_gt/total*100:.2f}%)")
    print(f"  Гарь (Prediction):   {burned_pred:,} ({burned_pred/total*100:.2f}%)")
    
    return cm, report

if __name__ == "__main__":
    print("🎯 Анализ матрицы ошибок\n")
    
    # Загрузка
    model = get_model(MODEL_PATH, DEVICE)
    print(f"✅ Модель загружена\n")
    
    # Dataloader
    val_folds = ['magenta']  # <-- Поменяй!
    
    # Создаём только val_loader
    all_folds = ['purple', 'coral', 'pink', 'grey', 'cyan', 'lime', 'magenta']
    train_folds_for_loader = [f for f in all_folds if f not in val_folds]

    _, val_loader = get_dataloaders(
        csv_path=CSV_PATH, base_dirs=BASE_DIRS,
        train_folds=train_folds_for_loader, val_folds=val_folds,
        batch_size=BATCH_SIZE, num_workers=2, img_size=IMG_SIZE
    )
    
    # Оценка
    cm, report = evaluate_confusion(model, val_loader, DEVICE)
    
    print("\n🎉 Анализ завершён!")
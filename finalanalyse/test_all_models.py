#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для сравнительного тестирования 4 моделей сегментации:
1. UNETR 3ch
2. UNETR 11ch
3. BasicUNet 3ch
4. BasicUNet 11ch

Требует наличия валидационных датасетов и чекпоинтов моделей.
"""

import os
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNETR, BasicUNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# === НАСТРОЙКИ ПУТЕЙ ===
# Путь к корневой папке с датасетами
BASE_DATASET_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset")

# Пути к валидационным папкам
VAL_DIR_3CH = BASE_DATASET_DIR / "validation_3ch"
VAL_DIR_11CH = BASE_DATASET_DIR / "validation_11ch"

# Пути к чекпоинтам моделей (взяты из ваших логов обучения)
CHECKPOINT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\NewTrain")
MODELS_CONFIG = {
    "UNETR 11ch": {
        "path": CHECKPOINT_DIR / "unetr_11ch_best.pth",
        "val_dir": VAL_DIR_11CH,
        "channels": 11,
        "type": "UNETR"
    },
    "BasicUNet 3ch": {
        "path": CHECKPOINT_DIR / "basicunet_3ch_best.pth",
        "val_dir": VAL_DIR_3CH,
        "channels": 3,
        "type": "BasicUNet"
    },
    "BasicUNet 11ch": {
        "path": CHECKPOINT_DIR / "basicunet_11ch_best.pth",
        "val_dir": VAL_DIR_11CH,
        "channels": 11,
        "type": "BasicUNet"
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# === DATASET CLASS ===
class ValidationDataset(Dataset):
    """Простой датасет для валидации без аугментаций"""
    def __init__(self, root_dir, channels=3):
        self.root_dir = Path(root_dir)
        self.channels = channels
        self.patches = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        
        if not self.patches:
            raise ValueError(f"Папка {root_dir} пуста или не содержит подпапок с патчами!")
            
        print(f"📂 Загружено {len(self.patches)} патчей из {root_dir.name}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_folder = self.patches[idx]
        
        # Определение имени файла изображения
        if self.channels == 3:
            img_path = patch_folder / "after_3ch.tif"
        else:
            # Для 11ch проверяем оба варианта, как в обучении
            img_path_11 = patch_folder / "after_11ch.tif"
            img_path_13 = patch_folder / "after_13ch.tif"
            if img_path_11.exists():
                img_path = img_path_11
            elif img_path_13.exists():
                img_path = img_path_13
            else:
                raise FileNotFoundError(f"Не найдено изображение в {patch_folder}")
                
        mask_path = patch_folder / "mask.tif"
        
        # Чтение
        image = tifffile.imread(img_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.float32)
        
        # Нормализация
        image = np.clip(image / 4000.0, 0, 1)
        mask = (mask > 0).astype(np.float32)
        
        # 🔧 ТРАНСПОНИРОВАНИЕ: (H, W, C) -> (C, H, W)
        if image.ndim == 3 and image.shape[0] != self.channels:
            # Если каналы последние, переставляем их на первое место
            image = np.transpose(image, (2, 0, 1))
            
        if mask.ndim == 3:
            mask = mask.squeeze()
            
        # Добавляем размерность канала для маски: (H, W) -> (1, H, W)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)

# === MODEL LOADING ===
def get_model(model_type, channels):
    if model_type == "UNETR":
        return UNETR(
            in_channels=channels, out_channels=1, img_size=(512, 512),
            spatial_dims=2, feature_size=16, norm_name="instance"
        )
    elif model_type == "BasicUNet":
        return BasicUNet(
            spatial_dims=2, in_channels=channels, out_channels=1,
            features=(32, 32, 64, 64, 128, 128), act="leakyrelu", norm="instance", dropout=0.0
        )

def evaluate_single_model(model_name, config):
    print(f"\n🚀 Тестирование: {model_name}...")
    
    # 1. Загрузка модели
    model = get_model(config["type"], config["channels"]).to(DEVICE)
    if not config["path"].exists():
        print(f"❌ Чекпоинт не найден: {config['path']}")
        return None
        
    checkpoint = torch.load(config["path"], map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Датасет и DataLoader
    dataset = ValidationDataset(config["val_dir"], config["channels"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. Инференс и сбор метрик
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > THRESHOLD).float()
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.cpu().numpy().flatten())
            
    # Конкатенация результатов
    y_true = np.concatenate(all_masks)
    y_pred = np.concatenate(all_preds)
    
    # 4. Расчет метрик
    # IoU (Jaccard)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    # Dice == F1 для бинарной задачи
    f1 = f1_score(y_true, y_pred, zero_division=0)
    dice = f1 
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        "IoU": iou,
        "Dice": dice,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "ConfusionMatrix": cm
    }
    
    print(f"✅ {model_name} завершена. IoU: {iou:.4f}, Dice: {dice:.4f}")
    return results

# === VISUALIZATION ===
def plot_results(all_results):
    models = list(all_results.keys())
    metrics = ["IoU", "Dice", "F1", "Precision", "Recall"]
    
    # Подготовка данных для графика
    data = []
    for model in models:
        for metric in metrics:
            data.append({"Model": model, "Metric": metric, "Value": all_results[model][metric]})
            
    import pandas as pd
    df = pd.DataFrame(data)
    
    # 1. Сравнительная столбчатая диаграмма
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Metric", y="Value", hue="Model", data=df, palette="viridis")
    plt.title("Сравнение метрик точности моделей", fontsize=16, fontweight='bold')
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Метрика", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Архитектура", fontsize=10)
    plt.tight_layout()
    plt.savefig("comparison_metrics.png", dpi=300)
    print("💾 Сохранено: comparison_metrics.png")
    plt.close()
    
    # 2. Матрицы ошибок (Confusion Matrices)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    if len(models) == 1:
        axes = [axes]
        
    for ax, model in zip(axes, models):
        cm = all_results[model]["ConfusionMatrix"]
        # Нормализация для отображения (опционально, здесь показываем абсолютные числа)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Background', 'Fire'], yticklabels=['Background', 'Fire'])
        ax.set_title(f"Матрица ошибок: {model}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Истинные значения", fontsize=10)
        ax.set_xlabel("Предсказанные значения", fontsize=10)
        
    plt.suptitle("Матрицы ошибок (Confusion Matrices)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300)
    print("💾 Сохранено: confusion_matrices.png")
    plt.close()

# === MAIN ===
def main():
    print("="*80)
    print("🔥 СРАВНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ МОДЕЛЕЙ СЕГМЕНТАЦИИ")
    print("="*80)
    print(f" Device: {DEVICE}")
    
    all_results = {}
    
    for model_name, config in MODELS_CONFIG.items():
        res = evaluate_single_model(model_name, config)
        if res:
            all_results[model_name] = res
            
    if not all_results:
        print("\n❌ Не удалось протестировать ни одну модель. Проверьте пути.")
        return
        
    print("\n" + "="*80)
    print(" СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    print(f"{'Модель':<15} | {'IoU':<6} | {'Dice':<6} | {'F1':<6} | {'Prec':<6} | {'Recall':<6}")
    print("-"*60)
    for model, res in all_results.items():
        print(f"{model:<15} | {res['IoU']:<6.4f} | {res['Dice']:<6.4f} | {res['F1']:<6.4f} | {res['Precision']:<6.4f} | {res['Recall']:<6.4f}")
    print("="*80)
    
    print("\n🎨 Генерация графиков...")
    plot_results(all_results)
    print("✅ Готово! Графики сохранены в текущей папке.")

if __name__ == "__main__":
    main()
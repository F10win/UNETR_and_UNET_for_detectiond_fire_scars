import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
from monai.networks.nets import UNETR
from dataset import SatelliteBurnDataset
import pandas as pd

# === НАСТРОЙКИ ===
MODEL_PATH = "checkpoints/unetr_best_improved.pth"
CSV_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\satellite_data.csv")
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
BASE_DIRS = [BASE_DIR / f"Satellite_burned_area_dataset_part{i}" for i in range(1, 6)]

IMG_SIZE = (512, 512)
IN_CHANNELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 5  # Сколько примеров показать

def get_model(model_path, device):
    model = UNETR(
        in_channels=IN_CHANNELS, out_channels=1, img_size=IMG_SIZE,
        spatial_dims=2, feature_size=16, norm_name="instance"
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def visualize_sample(img, pred, gt, folder_name, save_path):
    """Визуализация одного примера"""
    
    # Подготовка для отображения
    img_hwc = np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    
    # RGB каналы (False Color: 12,8,4)
    if img_hwc.shape[2] >= 12:
        rgb = img_hwc[:, :, [11, 7, 3]]  # SWIR2, NIR, Red
    else:
        rgb = img_hwc[:, :, :3]
    
    # Бинаризация
    pred_binary = (pred > 0.5).astype(np.float32)
    
    # Оверлеи
    overlay_pred = rgb.copy()
    overlay_pred[pred_binary > 0.5] = [1.0, 0.0, 0.0]  # Красный = предсказание
    
    overlay_gt = rgb.copy()
    overlay_gt[gt > 0.5] = [0.0, 1.0, 0.0]  # Зеленый = истина
    
    # Совпадения и ошибки
    overlay_errors = rgb.copy()
    overlay_errors[(pred_binary > 0.5) & (gt <= 0.5)] = [1.0, 0.5, 0.0]  # Оранжевый = False Positive
    overlay_errors[(pred_binary <= 0.5) & (gt > 0.5)] = [0.0, 0.5, 1.0]  # Голубой = False Negative
    
    # Метрики
    tp = np.sum((pred_binary > 0.5) & (gt > 0.5))
    fp = np.sum((pred_binary > 0.5) & (gt <= 0.5))
    fn = np.sum((pred_binary <= 0.5) & (gt > 0.5))
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Построение
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original (False Color: 12,8,4)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred, cmap='gray')
    axes[0, 1].set_title(f"Prediction\nProb > 0.5")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt, cmap='gray')
    axes[0, 2].set_title("Ground Truth")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(overlay_pred)
    axes[1, 0].set_title("Prediction Overlay (Red)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_gt)
    axes[1, 1].set_title("Ground Truth Overlay (Green)")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(overlay_errors)
    axes[1, 2].set_title(f"Errors (Orange=FP, Blue=FN)\nIoU: {iou:.3f}")
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Folder: {folder_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return iou

def main():
    print("🔍 Визуализация предсказаний\n")
    
    # Загрузка модели
    model = get_model(MODEL_PATH, DEVICE)
    print(f"✅ Модель загружена\n")
    
    # Загрузка датасета (валидационные fold'ы)
    val_folds = ['magenta']  # <-- Поменяй на нужный!
    
    dataset = SatelliteBurnDataset(
        base_dirs=BASE_DIRS,
        csv_path=CSV_PATH,
        fold_colors=val_folds,
        img_size=IMG_SIZE
    )
    
    print(f"📊 Примеров для визуализации: {min(NUM_SAMPLES, len(dataset))}\n")
    
    results = []
    
    with torch.no_grad():
        for idx in range(min(NUM_SAMPLES, len(dataset))):
            row = dataset.filtered_df.iloc[idx]
            folder_name = row['folder']
            
            # Загрузка данных
            folder_path = dataset._find_folder(folder_name)
            img = dataset._load_sentinel2(folder_path)
            gt = dataset._load_mask(folder_path)
            
            # Ресайз
            if img.shape[1:] != IMG_SIZE:
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
                img_tensor = TF.resize(img_tensor, IMG_SIZE, antialias=True)
                gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(DEVICE)
                gt_tensor = TF.resize(gt_tensor, IMG_SIZE, interpolation=TF.InterpolationMode.NEAREST)
                img = img_tensor.squeeze(0).cpu().numpy()
                gt = gt_tensor.squeeze(0).squeeze(0).cpu().numpy()
            
            # Предсказание
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Визуализация
            save_path = f"analysis/vis_sample_{idx:02d}_{folder_name[:30]}.png"
            iou = visualize_sample(img, pred, gt, folder_name, save_path)
            
            results.append({'folder': folder_name, 'IoU': iou})
            print(f"✅ [{idx+1}/{NUM_SAMPLES}] {folder_name[:40]}... IoU: {iou:.3f}")
    
    # Сводка
    if results:
        df = pd.DataFrame(results)
        print(f"\n📊 Средний IoU: {df['IoU'].mean():.3f} ± {df['IoU'].std():.3f}")
        print(f"📁 Визуализации сохранены в папке: analysis/")
    
    print("\n🎉 Визуализация завершена!")

if __name__ == "__main__":
    # Создаём папку для результатов
    Path("analysis").mkdir(exist_ok=True)
    main()
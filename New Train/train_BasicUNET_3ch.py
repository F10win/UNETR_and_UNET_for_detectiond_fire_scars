import os
import torch
import sys
import random
import logging
import numpy as np
import csv
import tifffile
import rasterio
from rasterio.transform import Affine
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch import nn
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss

from dataset_loader import get_dataloaders

# === НАСТРОЙКИ ===
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels")
CHECKPOINT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\NewTrain")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 📁 Путь для сохранения предсказаний (обновлён)
PRED_SAVE_DIR = Path(r"F:\Dataset\Basic UNET prediction\3 channels")
PRED_SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = 3
IMG_SIZE = (512, 512)
BATCH_SIZE = 1
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 5e-5
NUM_WORKERS = 1
MAX_PREDS_TO_SAVE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ЛОГИРОВАНИЕ ===
def setup_logging():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 📁 Обновлённый путь для логов
    log_dir = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\logs\NewTrain\BasicUNET\3channels")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_basicunet_3ch_{timestamp}.log"
    metrics_csv = log_dir / f"metrics_basicunet_3ch_{timestamp}.csv"
    
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    
    fmt = logging.Formatter('%(message)s')
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(fmt)
    root_logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler(log_file, encoding="utf-8")
    f_handler.setFormatter(fmt)
    root_logger.addHandler(f_handler)
    
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice', 'val_f1', 'lr'])
        
    return logging.getLogger(), metrics_csv

def get_model(in_channels=3):
    """Создание модели Basic UNet"""
    model = BasicUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        features=(32, 32, 64, 64, 128, 128),
        act="leakyrelu",
        norm="instance",
        dropout=0.0
    )
    return model

def calculate_metrics(outputs, masks):
    outputs_probs = torch.sigmoid(outputs)
    preds = (outputs_probs > 0.5).float()
    
    intersection = torch.sum(preds * masks)
    union = torch.sum(preds + masks) - intersection + 1e-6
    dice_num = 2 * intersection + 1e-6
    dice_denom = torch.sum(preds) + torch.sum(masks) + 1e-6
    
    iou = (intersection / union).item()
    dice = (dice_num / dice_denom).item()
    return iou, dice, dice

def save_georeferenced_prediction(pred_array, source_image_path, output_path):
    """
    Сохраняет предсказание с геопривязкой из исходного изображения
    
    Args:
        pred_array: numpy array предсказания (H, W) или (1, H, W)
        source_image_path: путь к исходному GeoTIFF для копирования метаданных
        output_path: путь для сохранения предсказания
    """
    try:
        # Читаем геопривязку из исходного изображения
        with rasterio.open(source_image_path, 'r') as src:
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            
            # Если pred_array имеет размерность (1, H, W), убираем лишний канал
            if pred_array.ndim == 3 and pred_array.shape[0] == 1:
                pred_array = pred_array[0]
            
            # Сохраняем предсказание с геопривязкой
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=pred_array.shape[0],
                width=pred_array.shape[1],
                count=1,
                dtype=pred_array.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress='lzw'
            ) as dst:
                dst.write(pred_array, 1)
                
        return True
    except Exception as e:
        logging.warning(f"⚠️  Не удалось сохранить геопривязку для {output_path}: {e}")
        # Если не получилось с геопривязкой, сохраняем обычный TIFF
        tifffile.imwrite(output_path, pred_array.astype(np.float32))
        return False

def main():
    logger, metrics_csv = setup_logging()
    
    # ✅ Оптимизация GPU + воспроизводимость
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    
    logger.info("="*70)
    logger.info("🔷 Обучение Basic UNet на 3 каналах (80/20 сплит)")
    logger.info(f"📂 Датасет: {BASE_DIR}")
    logger.info(f"📐 Размер: {IMG_SIZE} | 🔢 Каналов: {CHANNELS}")
    logger.info(f"📦 Batch Size: {BATCH_SIZE} | 👷 Workers: {NUM_WORKERS}")
    logger.info(f"💾 Предсказания с геопривязкой: {PRED_SAVE_DIR}")
    logger.info("⚡ Режим: AMP (Mixed Precision)")
    logger.info("="*70)
    
    # 🔀 Разделение 80/20 с полным перемешиванием
    all_patches = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
    total_count = len(all_patches)
    indices = list(range(total_count))
    random.shuffle(indices)
    
    split_idx = int(0.8 * total_count)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    logger.info(f"📊 Всего патчей: {total_count}")
    logger.info(f"🔹 Train: {len(train_indices)} | 🟦 Val: {len(val_indices)}")
    
    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        base_path=BASE_DIR, channels='3', batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, train_indices=train_indices,
        val_indices=val_indices, augment=True
    )
    
    # Модель: Basic UNet
    model = get_model(in_channels=CHANNELS).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📦 Параметры модели: {total_params:,} (trainable: {trainable_params:,})")
    
    criterion = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, smooth_nr=1.0, smooth_dr=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # ✅ GradScaler с правильным API
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = CHECKPOINT_DIR / "basicunet_3ch_best.pth"
    start_epoch = 0
    
    # Загрузка чекпоинта
    if best_model_path.exists():
        logger.info(f"🔄 Загрузка чекпоинта: {best_model_path}")
        try:
            ckpt = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for pg in optimizer.param_groups: pg['lr'] = LEARNING_RATE
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['val_loss']
            logger.info(f"📅 Продолжаем с Эпохи: {start_epoch} | Лучший Loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки: {e}. Начинаем с нуля.")
    else:
        logger.info("📁 Чекпоинт не найден, начинаем с нуля")
            
    logger.info(f"🚀 Начало обучения Basic UNet на {DEVICE} (AMP)\n")
    
    with open(metrics_csv, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        for epoch in range(start_epoch, MAX_EPOCHS):
            # === TRAIN ===
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]", leave=False)
            
            for images, masks in pbar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            train_loss /= len(train_loader)
            
            # === VALIDATION ===
            model.eval()
            val_loss = 0.0
            val_iou, val_dice, val_f1 = 0.0, 0.0, 0.0
            valid_batches = 0
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for images, masks in val_loader:
                        images, masks = images.to(DEVICE), masks.to(DEVICE)
                        outputs = model(images)
                        batch_loss = criterion(outputs, masks)
                        
                        if not torch.isnan(batch_loss):
                            val_loss += batch_loss.item()
                            iou, dice, f1 = calculate_metrics(outputs, masks)
                            val_iou += iou
                            val_dice += dice
                            val_f1 += f1
                            valid_batches += 1
                        
            if valid_batches > 0:
                val_loss /= valid_batches
                val_iou /= valid_batches
                val_dice /= valid_batches
                val_f1 /= valid_batches
            else:
                val_loss = float('inf')
                
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = (f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                       f"IoU: {val_iou:.4f} | Dice/F1: {val_dice:.4f} | LR: {current_lr:.6f}")
            logger.info(log_msg)
            
            csv_writer.writerow([epoch+1, round(train_loss, 6), round(val_loss, 6),
                                 round(val_iou, 6), round(val_dice, 6), round(val_f1, 6),
                                 round(current_lr, 8)])
                                 
            # === CHECKPOINT & EARLY STOPPING + СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ ===
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': {'channels': CHANNELS, 'img_size': IMG_SIZE, 'architecture': 'BasicUNet'}
                }, best_model_path)
                logger.info(f"✅ Новый рекорд! Сохранено: {best_model_path.name}")
                
                # 📸 Сохранение предсказаний С ГЕОПРИВЯЗКОЙ
                epoch_pred_dir = PRED_SAVE_DIR / f"epoch_{epoch+1}_loss_{val_loss:.4f}"
                epoch_pred_dir.mkdir(parents=True, exist_ok=True)
                
                saved_count = 0
                # Получаем индексы валидации для доступа к путям файлов
                val_patches = [all_patches[i] for i in val_indices if i < len(all_patches)]
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        for batch_idx, (images, masks) in enumerate(val_loader):
                            if saved_count >= MAX_PREDS_TO_SAVE:
                                break
                                
                            images, masks = images.to(DEVICE), masks.to(DEVICE)
                            outputs = model(images)
                            probs = torch.sigmoid(outputs)
                            
                            # 🌍 Сохраняем предсказание с геопривязкой
                            pred_array = probs[0, 0].cpu().numpy().astype(np.float32)
                            gt_array = masks[0, 0].cpu().numpy().astype(np.float32)
                            
                            # Получаем путь к исходному изображению для геопривязки
                            if batch_idx < len(val_patches):
                                source_patch = val_patches[batch_idx]
                                source_image_path = source_patch / "after_3ch.tif"
                                if not source_image_path.exists():
                                    source_image_path = source_patch / "after_11ch.tif"
                                
                                if source_image_path.exists():
                                    # Сохраняем предсказание с геопривязкой
                                    pred_path = epoch_pred_dir / f"pred_{saved_count:03d}.tif"
                                    gt_path = epoch_pred_dir / f"gt_{saved_count:03d}.tif"
                                    
                                    save_georeferenced_prediction(pred_array, source_image_path, pred_path)
                                    save_georeferenced_prediction(gt_array, source_image_path, gt_path)
                                    
                                    logger.info(f"  💾 Сохранено: pred_{saved_count:03d}.tif (GeoTIFF)")
                                else:
                                    logger.warning(f"  ⚠️  Исходный файл не найден: {source_image_path}")
                            else:
                                # Если не нашли исходный файл, сохраняем без геопривязки
                                tifffile.imwrite(epoch_pred_dir / f"pred_{saved_count:03d}.tif", pred_array)
                                tifffile.imwrite(epoch_pred_dir / f"gt_{saved_count:03d}.tif", gt_array)
                            
                            saved_count += 1
                            
                logger.info(f"📸 Сохранено {saved_count} предсказаний с геопривязкой в: {epoch_pred_dir}")
                
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    logger.info(f"\n⏹️ Early Stopping! Нет улучшений {PATIENCE} эпох.")
                    logger.info(f"🏆 Лучший Val Loss: {best_val_loss:.4f}")
                    break
                    
    logger.info("\n" + "="*70)
    logger.info("🎉 ОБУЧЕНИЕ BASIC UNET ЗАВЕРШЕНО!")
    logger.info(f"📁 Лучшая модель: {best_model_path}")
    logger.info(f"📊 Метрики: {metrics_csv}")
    logger.info(f"📸 Маски с геопривязкой: {PRED_SAVE_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\n❌ Прервано пользователем")
    except Exception as e:
        logging.error(f"\n\n❌ Ошибка: {e}")
        import traceback
        logging.error(traceback.format_exc())
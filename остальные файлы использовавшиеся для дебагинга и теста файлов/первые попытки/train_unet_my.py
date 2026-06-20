#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение стандартной U-Net на патчах Sentinel-2
Оптимизировано под скорость и стабильность
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import BasicUNet  # Классическая архитектура U-Net
from monai.losses import DiceLoss
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import warnings
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# ⚙️ === КОНФИГУРАЦИЯ (МЕНЯЙ ЗДЕСЬ) ===
CONFIG = {
    'channels': 11,  # 3 для SWIR/NIR/Red или 11 для мультиспектра
    'dataset_path': Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\13_channels"),
    # Если 11 каналов, укажи: Path(r"E:\...\dataset\13_channels")
    
    'batch_size': 2,  # 2 для 3ch, 1 для 11ch (из-за VRAM)
    'max_epochs': 100,
    'early_stopping_patience': 10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'train_ratio': 0.8,  # 80% train, 20% val
    'random_state': 42,
    
    'checkpoint_dir': Path("checkpoints_unet"),
    'log_dir': Path("logs_unet"),
}

# Создание папок
CONFIG['checkpoint_dir'].mkdir(exist_ok=True)
CONFIG['log_dir'].mkdir(exist_ok=True)

# 🔍 Логирование в файл с меткой времени
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = CONFIG['log_dir'] / f"unet_training_log_{timestamp}.txt"
sys.stdout = open(log_file, "w", encoding="utf-8")

# 📦 Dataset
class BurnDataset(Dataset):
    def __init__(self, patch_names, base_path, transform=None):
        self.patch_names = patch_names
        self.base_path = Path(base_path)
        self.transform = transform

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch_folder = self.base_path / self.patch_names[idx]
        
        # Выбираем файл в зависимости от каналов
        img_name = "after_13ch.tif" if CONFIG['channels'] == 11 else "after_3ch.tif"
        img = tifffile.imread(patch_folder / img_name).astype(np.float32)
        mask = tifffile.imread(patch_folder / "mask.tif").astype(np.float32)

        # Нормализация
        img = np.clip(img / 4000.0, 0, 1)
        mask = (mask > 0).astype(np.float32)

        # Albumentations требует формат (H, W, C)
        if img.ndim == 3 and img.shape[0] == CONFIG['channels']:
            img = np.transpose(img, (1, 2, 0))

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        # Маска в формате (1, H, W)
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        return img, mask

# 🎨 Аугментации
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.0]*CONFIG['channels'], std=[1.0]*CONFIG['channels']),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    
    return A.Compose([
        A.Normalize(mean=[0.0]*CONFIG['channels'], std=[1.0]*CONFIG['channels']),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

# 🚀 Обучение
def train():
    print("="*70)
    print(f"🔥 ОБУЧЕНИЕ U-Net ({CONFIG['channels']} канала)")
    print("="*70)
    print(f"📂 Датасет: {CONFIG['dataset_path']}")
    print(f"📦 Batch Size: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"🛑 Early Stopping: {CONFIG['early_stopping_patience']} эпох")
    print("="*70)

    # Разделение 80/20
    all_patches = sorted([p.name for p in CONFIG['dataset_path'].iterdir() if p.is_dir()])
    random.seed(CONFIG['random_state'])
    random.shuffle(all_patches)
    split_idx = int(len(all_patches) * CONFIG['train_ratio'])
    train_patches = all_patches[:split_idx]
    val_patches = all_patches[split_idx:]
    print(f"📊 Train: {len(train_patches)} | Val: {len(val_patches)}\n")

    # DataLoaders (оптимизированы под GPU)
    train_ds = BurnDataset(train_patches, CONFIG['dataset_path'], get_transforms(True))
    val_ds = BurnDataset(val_patches, CONFIG['dataset_path'], get_transforms(False))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=2, pin_memory=True)

    # Модель (Классическая U-Net из MONAI)
    model = BasicUNet(
        spatial_dims=2,
        in_channels=CONFIG['channels'],
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32)
    ).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Параметры U-Net: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Loss, Optimizer, Scheduler
    criterion = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = CONFIG['checkpoint_dir'] / f"unet_{CONFIG['channels']}ch_best.pth"

    print("\n🏁 Начало обучения...")
    for epoch in range(1, CONFIG['max_epochs'] + 1):
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['max_epochs']} [Train]", leave=False)
        
        for imgs, masks in pbar:
            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Быстрее обнуление
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Защита от взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

        # === CHECKPOINT & EARLY STOPPING ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG
            }, best_model_path)
            print(f"✅ Новый рекорд! Сохранено: {best_model_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⏹️ Early Stopping! Нет улучшений {CONFIG['early_stopping_patience']} эпох.")
                print(f"🏆 Лучший Val Loss: {best_val_loss:.4f} (эпоха {epoch - patience_counter})")
                break

    print("\n🎉 Обучение U-Net завершено!")
    print(f"📁 Лучшая модель: {best_model_path}")
    print(f"📜 Лог сохранён: {log_file}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tifffile
import random
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BurnedAreaDataset(Dataset):
    """Dataset для загрузки патчей с пожарами"""
    
    def __init__(self, base_path, channels='3', augment=False, patch_indices=None):
        """
        Args:
            base_path: Путь к папке с датасетом (13_channels или 3_channels)
            channels: '11' для 11 каналов, '3' для 3 каналов
            augment: Применять аугментации
            patch_indices: Список индексов патчей для использования (для CV)
        """
        self.base_path = Path(base_path)
        self.channels = channels
        self.augment = augment
        
        # Получаем список всех папок с патчами
        all_patches = sorted([p for p in self.base_path.iterdir() if p.is_dir()])
        
        # Если указаны индексы, используем только их
        if patch_indices is not None:
            self.patches = [all_patches[i] for i in patch_indices if i < len(all_patches)]
        else:
            self.patches = all_patches
        
        print(f"📦 Загружено {len(self.patches)} патчей из {self.base_path}")
        
        # Аугментации
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_folder = self.patches[idx]
    
        # Определяем имя файла в зависимости от каналов
        if self.channels == '11' or self.channels == '13':
            image_path = patch_folder / "after_13ch.tif"
        else:  # '3' канала
            image_path = patch_folder / "after_3ch.tif"
    
        mask_path = patch_folder / "mask.tif"
    
        # Чтение TIFF
        image = tifffile.imread(image_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.float32)
    
        # Нормализация
        image = np.clip(image / 4000.0, 0, 1)
    
        # Бинаризация маски
        mask = (mask > 0).astype(np.float32)
    
        # Приводим к формату (H, W, C) для albumentations
        if image.ndim == 3 and image.shape[0] < 20:
            image = np.transpose(image, (1, 2, 0))
    
        # Маска должна быть (H, W)
        if mask.ndim == 3:
            mask = mask.squeeze()
    
        # Применяем аугментации
        augmented = self.transform(image=image, mask=mask)
    
        image = augmented['image']  # (C, H, W)
        mask = augmented['mask']    # (H, W)
    
        # 🔥 ДОБАВЬ ЭТУ СТРОКУ - добавляем канал к маске!
        mask = torch.unsqueeze(mask, 0)  # (H, W) -> (1, H, W)
    
        return image, mask


def get_cross_validation_splits(dataset_size, n_splits=5, random_state=42):
    """Генерация индексов для k-fold кросс-валидации"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_idx, val_idx in kfold.split(range(dataset_size)):
        splits.append({
            'train': train_idx.tolist(),
            'val': val_idx.tolist()
        })
    return splits


def get_dataloaders(base_path, channels='3', batch_size=2, num_workers=2, 
                    train_indices=None, val_indices=None, augment=True):
    """Создание DataLoader для train и val"""
    
    train_dataset = BurnedAreaDataset(
        base_path=base_path,
        channels=channels,
        augment=augment,
        patch_indices=train_indices
    )
    
    val_dataset = BurnedAreaDataset(
        base_path=base_path,
        channels=channels,
        augment=False,
        patch_indices=val_indices
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
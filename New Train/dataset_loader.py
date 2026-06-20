import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BurnedAreaDataset(Dataset):
    """Dataset для загрузки патчей с пожарами"""
    def __init__(self, base_path, channels='11', augment=False, patch_indices=None):
        self.base_path = Path(base_path)
        self.channels = channels
        self.augment = augment
        
        all_patches = sorted([p for p in self.base_path.iterdir() if p.is_dir()])
        if patch_indices is not None:
            self.patches = [all_patches[i] for i in patch_indices if i < len(all_patches)]
        else:
            self.patches = all_patches
            
        print(f"📦 Загружено {len(self.patches)} патчей из {self.base_path}")
        
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    cval=0,
                    p=0.5
                ),
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
        
        # 🔍 Пробуем найти файл с изображением (поддержка обоих имён)
        image_path_11ch = patch_folder / "after_11ch.tif"
        image_path_13ch = patch_folder / "after_13ch.tif"
        
        if image_path_11ch.exists():
            image_path = image_path_11ch
        elif image_path_13ch.exists():
            image_path = image_path_13ch
        else:
            raise FileNotFoundError(f"Не найдено изображение в {patch_folder}. "
                                    f"Ожидалось: after_13ch.tif или after_11ch.tif")
        
        mask_path = patch_folder / "mask.tif"
        
        # 🛡️ Обработка ошибок чтения TIFF
        try:
            image = tifffile.imread(image_path).astype(np.float32)
            mask = tifffile.imread(mask_path).astype(np.float32)
        except Exception as e:
            print(f"⚠️  Ошибка чтения файла {image_path}: {e}")
            image = np.zeros((512, 512, 3), dtype=np.float32)
            mask = np.zeros((512, 512), dtype=np.float32)
        
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
        
        mask = torch.unsqueeze(mask, 0)  # (H, W) -> (1, H, W)
        
        return image, mask


def get_dataloaders(base_path, channels='11', batch_size=1, num_workers=2, 
                    train_indices=None, val_indices=None, augment=True):
    train_dataset = BurnedAreaDataset(base_path, channels, augment=augment, patch_indices=train_indices)
    val_dataset = BurnedAreaDataset(base_path, channels, augment=False, patch_indices=val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import tifffile

class SatelliteBurnDataset(Dataset):
    """
    Датасет для Satellite Burned Area Dataset
    """
    def __init__(self, base_dirs, csv_path, fold_colors, img_size=(512, 512)):
        self.base_dirs = [Path(d) for d in base_dirs]
        self.csv_path = Path(csv_path)
        self.fold_colors = fold_colors
        self.img_size = img_size
        
        # Читаем CSV с правильным разделителем
        self.df = pd.read_csv(csv_path, sep=';')
        
        # Фильтруем по fold'ам
        self.filtered_df = self.df[self.df['fold'].isin(fold_colors)].reset_index(drop=True)
        print(f"✅ Загружено {len(self.filtered_df)} примеров для fold'ов: {fold_colors}")
    
    def __len__(self):
        return len(self.filtered_df)
    
    def _find_folder(self, folder_name):
        """Поиск папки с данными во всех частях датасета"""
        for base_dir in self.base_dirs:
            folder_path = base_dir / folder_name
            if folder_path.exists():
                return folder_path
        raise FileNotFoundError(f"Папка {folder_name} не найдена")
    
    def _load_sentinel2(self, folder_path):
        """Загрузка ОДНОГО файла Sentinel-2 (12 каналов)"""
        # Ищем все файлы Sentinel-2
        s2_files = sorted(glob.glob(str(folder_path / "sentinel2_*.tiff")))
        if not s2_files:
            raise FileNotFoundError(f"Sentinel-2 файлы не найдены в {folder_path}")
        
        # Берём ПОСЛЕДНИЙ файл (post-fire снимок)
        s2_file = s2_files[-1]
        
        try:
            # Используем tifffile для чтения GeoTIFF
            img = tifffile.imread(s2_file).astype(np.float32)
            
            # tifffile возвращает (H, W, C) — каналы последние
            if img.ndim == 3:
                # Переводим в (C, H, W) для PyTorch
                img = np.transpose(img, (2, 0, 1))
            else:
                # 2D изображение — добавляем измерение канала
                img = np.expand_dims(img, 0)
            
            # Проверяем количество каналов
            n_channels = img.shape[0]
            print(f"  📊 {folder_path.name}: {n_channels} каналов, размер {img.shape[1:]}")
            
            # Если каналов < 12, дополняем нулями
            if n_channels < 12:
                padding = np.zeros((12 - n_channels, *img.shape[1:]), dtype=np.float32)
                img = np.concatenate([img, padding], axis=0)
            elif n_channels > 12:
                img = img[:12]
            
            # Нормализация (данные уже в диапазоне 0-1)
            img = np.clip(img, 0, 1)
            
            return img
            
        except Exception as e:
            raise FileNotFoundError(f"Не удалось прочитать {s2_file.name}: {e}")
    
    def _load_mask(self, folder_path):
        """Загрузка маски"""
        mask_files = list(glob.glob(str(folder_path / "*_mask.tiff"))) + \
                     list(glob.glob(str(folder_path / "*_mask.png")))
        
        if not mask_files:
            raise FileNotFoundError(f"Маска не найдена в {folder_path}")
        
        mask_file = mask_files[0]
        
        # Используем tifffile для TIFF, PIL для PNG
        if mask_file.endswith('.tiff') or mask_file.endswith('.tif'):
            mask = tifffile.imread(mask_file).astype(np.float32)
        else:
            mask = np.array(Image.open(mask_file)).astype(np.float32)
        
        # Бинаризация: 0-36 = не гарь, 37-255 = гарь
        if mask.max() > 1:
            mask = (mask >= 37).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        return mask
    
    def __getitem__(self, idx):
        row = self.filtered_df.iloc[idx]
        folder_name = row['folder']
        
        folder_path = self._find_folder(folder_name)
        
        # Загружаем данные
        img = self._load_sentinel2(folder_path)  # (12, H, W)
        mask = self._load_mask(folder_path)      # (H, W)
        
        # Ресайз до целевого размера (ОБЯЗАТЕЛЬНО!)
        if img.shape[1:] != self.img_size:
            img_tensor = torch.from_numpy(img)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            img_tensor = TF.resize(img_tensor, self.img_size, antialias=True)
            mask_tensor = TF.resize(mask_tensor, self.img_size, 
                                   interpolation=TF.InterpolationMode.NEAREST)
            img = img_tensor.numpy()
            mask = mask_tensor.squeeze(0).numpy()
        
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0)


def get_dataloaders(csv_path, base_dirs, train_folds, val_folds, 
                   batch_size=4, num_workers=2, img_size=(512, 512)):
    """Создание dataloader'ов"""
    
    # Train с аугментациями
    train_dataset = SatelliteBurnDataset(
        base_dirs=base_dirs,
        csv_path=csv_path,
        fold_colors=train_folds,
        img_size=img_size
    )
    
    # Val без аугментаций
    val_dataset = SatelliteBurnDataset(
        base_dirs=base_dirs,
        csv_path=csv_path,
        fold_colors=val_folds,
        img_size=img_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, 
                           pin_memory=True)
    
    return train_loader, val_loader
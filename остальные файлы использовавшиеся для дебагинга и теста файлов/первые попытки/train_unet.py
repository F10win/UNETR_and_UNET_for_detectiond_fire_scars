import os
import torch
from torch import nn
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from dataset import get_dataloaders
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

def get_model(img_size, in_channels=12, out_channels=1, device="cpu"):
    """Создание модели UNet"""
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),  # Глубина сети
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch"
    )
    return model.to(device)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Обучение UNet на устройстве: {device}")
    
    # === ПУТИ К ДАННЫМ ===
    BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
    CSV_PATH = BASE_DIR / "satellite_data.csv"
    
    BASE_DIRS = [
        BASE_DIR / "Satellite_burned_area_dataset_part1",
        BASE_DIR / "Satellite_burned_area_dataset_part2",
        BASE_DIR / "Satellite_burned_area_dataset_part3",
        BASE_DIR / "Satellite_burned_area_dataset_part4",
        BASE_DIR / "Satellite_burned_area_dataset_part5",
    ]
    
    # === ПАРАМЕТРЫ ===
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4  # UNet легче, можно больше batch size
    EPOCHS = 50
    IN_CHANNELS = 12
    
    # === FOLDS ===
    TRAIN_FOLDS = ['purple', 'coral', 'pink', 'grey', 'cyan', 'lime']
    VAL_FOLDS = ['magenta']
    
    print(f"📊 Обучение на fold'ах: {TRAIN_FOLDS}")
    print(f"📊 Валидация на fold'ах: {VAL_FOLDS}\n")
    
    # Создание dataloader'ов
    train_loader, val_loader = get_dataloaders(
        csv_path=CSV_PATH,
        base_dirs=BASE_DIRS,
        train_folds=TRAIN_FOLDS,
        val_folds=VAL_FOLDS,
        batch_size=BATCH_SIZE,
        num_workers=2,
        img_size=IMG_SIZE
    )
    
    # Модель
    model = get_model(img_size=IMG_SIZE, in_channels=IN_CHANNELS,
                     out_channels=1, device=device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Всего параметров: {total_params:,}")
    print(f"📦 Обучаемых параметров: {trainable_params:,}\n")
    
    # DiceFocalLoss
    criterion = DiceFocalLoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True,
        squared_pred=True,
        jaccard=True,
        alpha=0.5,
        gamma=2.0,
        smooth_nr=1e-5,
        smooth_dr=1e-5
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # GradScaler для AMP
    scaler = GradScaler()
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    
    print(f"📈 Начало обучения: {EPOCHS} эпох (с DiceFocalLoss + AMP)")
    print(f"📦 Train samples: {len(train_loader.dataset)}")
    print(f"📦 Val samples: {len(val_loader.dataset)}\n")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            # AMP (Automatic Mixed Precision)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    val_loss += criterion(model(images), masks).item()
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_folds': TRAIN_FOLDS,
                'val_folds': VAL_FOLDS,
                'model_name': 'UNet'
            }, "checkpoints/unet_best.pth")
            print("✅ Сохранена лучшая UNet модель.\n")
            
    print("\n🎉 Обучение UNet завершено!")
    print(f"🏆 Лучший val loss: {best_val_loss:.4f}")
    
    # Сравнение с UNETR
    print("\n" + "="*60)
    print("📊 СРАВНЕНИЕ МОДЕЛЕЙ:")
    print("="*60)
    print(f"{'Модель':<15} {'Параметры':<15} {'Val Loss':<15}")
    print("-"*60)
    print(f"{'UNet':<15} {trainable_params:<15,} {best_val_loss:<15.4f}")
    print("="*60)

if __name__ == "__main__":
    train()
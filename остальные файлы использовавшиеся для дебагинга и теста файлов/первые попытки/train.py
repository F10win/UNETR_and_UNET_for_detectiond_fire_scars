import os
import torch
from torch import nn
from tqdm import tqdm
from monai.networks.nets import UNETR
from monai.losses import DiceLoss
from dataset import get_dataloaders
from pathlib import Path

def get_model(img_size, in_channels=13, out_channels=1, device="cpu"):
    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        spatial_dims=2,
        feature_size=16,
        norm_name="instance"
    )
    return model.to(device)

class CombinedLoss(nn.Module):
    """Комбинация BCE и Dice Loss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        # Пробуем разные варианты параметров для совместимости
        try:
            # Новая версия MONAI
            self.dice = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
        except TypeError:
            try:
                # Старая версия MONAI
                self.dice = DiceLoss(sigmoid=True, smooth=1e-5)
            except TypeError:
                # Минимальная версия без параметров
                self.dice = DiceLoss(sigmoid=True)
    
    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

def train(train_folds, val_folds, result_path="checkpoints/unetr_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Устройство: {device}")
    
    # ТВОЙ ПУТЬ
    BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
    CSV_PATH = BASE_DIR / "satellite_data.csv"
    BASE_DIRS = [BASE_DIR / f"Satellite_burned_area_dataset_part{i}" for i in range(1, 6)]
    
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 2
    EPOCHS = 40 # Уменьшим, добавим Early Stopping
    
    train_loader, val_loader = get_dataloaders(
        csv_path=CSV_PATH, base_dirs=BASE_DIRS,
        train_folds=train_folds, val_folds=val_folds,
        batch_size=BATCH_SIZE, num_workers=2, img_size=IMG_SIZE
    )
    
    model = get_model(img_size=IMG_SIZE, in_channels=13, out_channels=1, device=device)
    
    # Используем новую функцию потерь
    criterion = CombinedLoss() 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = 10 # Early Stopping
    
    print(f"📈 Обучение на: {train_folds}, Валидация на: {val_folds}\n")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_loss += criterion(model(images), masks).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss) # Уменьшаем LR если нет улучшений
        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_folds': train_folds,
                'val_folds': val_folds
            }, result_path)
            print("💾 Сохранено!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⏹️ Early Stopping на эпохе {epoch+1}")
                break

    return best_val_loss

if __name__ == "__main__":
    # Тестовый запуск для 1 fold
    TRAIN_FOLDS = ['purple', 'coral', 'pink', 'grey', 'cyan', 'lime']
    VAL_FOLDS = ['magenta']
    train(TRAIN_FOLDS, VAL_FOLDS)
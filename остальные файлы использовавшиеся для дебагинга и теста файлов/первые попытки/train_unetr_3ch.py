import os
import torch
import sys
from datetime import datetime
from torch import nn
from tqdm import tqdm
from monai.networks.nets import UNETR
from monai.losses import DiceLoss  # ← Используем простой DiceLoss вместо DiceFocalLoss
from pathlib import Path
import numpy as np

from dataset_loader import get_dataloaders, get_cross_validation_splits

# === НАСТРОЙКИ ===
BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# === АВТОМАТИЧЕСКОЕ ЛОГИРОВАНИЕ ===
def start_logging(model_name="3ch"):
    # Создаём папку logs/{model_name}
    log_dir = Path("logs") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Уникальное имя файла с датой и временем
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    
    # Перенаправляем весь вывод в файл
    sys.stdout = open(log_path, "w", encoding="utf-8")
    
    print(f"📝 Логирование запущено")
    print(f"📁 Путь: {log_path}")
    print(f"🕐 Время: {timestamp}")
    print("="*60)
    
    return log_path

# Запускаем логирование
log_file = start_logging(model_name="3ch")  # или "11ch" для другой модели
# =======================================

# Параметры
CHANNELS = 3  # 3 канала (SWIR=B12, NIR=B8, Red=B4)
IMG_SIZE = (512, 512)
BATCH_SIZE = 1  # ← Уменьшил до 1 для стабильности (8 GB VRAM)
MAX_EPOCHS = 100
PATIENCE = 10  # Early stopping
LEARNING_RATE = 5e-5

# Кросс-валидация
N_FOLDS = 2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(in_channels=3):
    """Создание модели UNETR"""
    model = UNETR(
        in_channels=in_channels,
        out_channels=1,
        img_size=IMG_SIZE,
        spatial_dims=2,
        feature_size=16,
        norm_name="instance"
    )
    return model

def train_fold(fold_num, train_indices, val_indices):
    """Обучение одного fold'а"""
    print(f"\n{'='*70}")
    print(f"🔁 FOLD {fold_num + 1}/{N_FOLDS}")
    print(f"📊 Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"{'='*70}\n")
    
    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        base_path=BASE_DIR,
        channels='3',
        batch_size=BATCH_SIZE,
        num_workers=2,
        train_indices=train_indices,
        val_indices=val_indices,
        augment=True
    )
    
    # Модель
    model = get_model(in_channels=CHANNELS).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Параметры модели: {total_params:,}")
    
    # ← ИСПОЛЬЗУЕМ DiceLoss ВМЕСТО DiceFocalLoss (стабильнее)
    criterion = DiceLoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True,
        smooth_nr=1.0,  # ← Увеличил для стабильности
        smooth_dr=1.0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Scheduler (адаптивное снижение LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # ← УБРАЛ GradScaler (нет AMP)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = CHECKPOINT_DIR / f"unetr_3ch_fold{fold_num+1}.pth"
    
    print(f"🚀 Начало обучения на {DEVICE} (FP32, без AMP)\n")
    
    # ← ЗАГРУЗКА ЧЕКПОИНТА С ЭПОХИ 26 (последний хороший рекорд)
    checkpoint_path = CHECKPOINT_DIR / "unetr_3ch_fold1.pth"
    start_epoch = 0
    
    if checkpoint_path.exists():
        print(f"🔄 Загрузка чекпоинта: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Принудительно устанавливаем новый LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"📅 Продолжаем с Эпохи: {start_epoch}")
            print(f"🔧 Learning Rate установлен: {LEARNING_RATE}")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить чекпоинт: {e}")
            print("🔄 Начинаем обучение с нуля")
            start_epoch = 0
    else:
        print("📁 Чекпоинт не найден, начинаем с нуля")

    # Цикл обучения:
    for epoch in range(start_epoch, MAX_EPOCHS):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold_num+1} | Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
    
            # ← УБРАЛ autocast (чистый FP32)
            outputs = model(images)
            loss = criterion(outputs, masks)
        
            # 🛡️ ПРОВЕРКА НА NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  [WARNING] Пропускаю битый батч (NaN/Inf loss={loss.item()})")
                continue
            
            # ← УБРАЛ scaler (нет AMP)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                # ← УБРАЛ autocast
                outputs = model(images)
                batch_loss = criterion(outputs, masks)
                
                if not torch.isnan(batch_loss):
                    val_loss += batch_loss.item()
                    valid_batches += 1
        
        if valid_batches > 0:
            val_loss /= valid_batches
        else:
            val_loss = float('inf')
            print("⚠️  [WARNING] Все валидационные батчи содержат NaN!")
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nFold {fold_num+1} | Epoch {epoch+1:02d} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # ========== CHECKPOINTING ==========
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            torch.save({
                'fold': fold_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'channels': CHANNELS,
                'config': {
                    'img_size': IMG_SIZE,
                    'in_channels': CHANNELS,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE
                }
            }, best_model_path)
            
            print(f"✅ Новый рекорд! Сохранено: {best_model_path.name}\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n⏹️ Early Stopping! Нет улучшений {PATIENCE} эпох.")
                print(f"🏆 Лучший Val Loss: {best_val_loss:.4f} (эпоха {epoch-PATIENCE+1})")
                break
    
    return best_val_loss

def main():
    print("="*70)
    print("🛰️  Обучение UNETR на 3 каналах (SWIR, NIR, Red)")
    print("="*70)
    print(f"\n📂 Датасет: {BASE_DIR}")
    print(f"📐 Размер: {IMG_SIZE}")
    print(f"🔢 Каналов: {CHANNELS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Folds: {N_FOLDS}")
    print(f"⏱️  Max Epochs: {MAX_EPOCHS}")
    print(f"🛑 Early Stopping Patience: {PATIENCE}")
    print(f"💾 Checkpoints: {CHECKPOINT_DIR}")
    print(f"🔧 Режим: FP32 (без AMP), DiceLoss")
    print("="*70)
    
    # Генерация splits для кросс-валидации
    print("\n🔄 Генерация fold'ов для кросс-валидации...")
    dataset_size = 2386
    splits = get_cross_validation_splits(dataset_size, n_splits=N_FOLDS, random_state=RANDOM_STATE)
    
    print(f"✅ Создано {len(splits)} fold'ов\n")
    
    # Обучение на каждом fold'е
    fold_results = []
    for fold_num, split in enumerate(splits):
        best_loss = train_fold(
            fold_num=fold_num,
            train_indices=split['train'],
            val_indices=split['val']
        )
        fold_results.append(best_loss)
    
    # Итоговая статистика
    print("\n" + "="*70)
    print("🎉 КРОСС-ВАЛИДАЦИЯ ЗАВЕРШЕНА!")
    print("="*70)
    print(f"\n📊 Результаты по fold'ам:")
    for i, loss in enumerate(fold_results):
        print(f"   Fold {i+1}: Val Loss = {loss:.4f}")
    
    print(f"\n📈 Средний Val Loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"📁 Все модели сохранены в: {CHECKPOINT_DIR}")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
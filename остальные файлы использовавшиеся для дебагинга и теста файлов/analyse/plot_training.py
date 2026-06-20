import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Путь к чекпоинтам
CHECKPOINT_DIR = Path("checkpoints")

def load_training_history(checkpoint_path):
    """Загружает историю из checkpoint (если сохранена)"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except:
        return None

def plot_loss_curves(train_losses, val_losses, save_path="training_loss.png"):
    """Строит график потерь"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Динамика потерь во время обучения', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ График сохранён: {save_path}")
    plt.show()

def analyze_checkpoint(checkpoint_path):
    """Анализ одного checkpoint файла"""
    print(f"🔍 Анализ: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    epoch = checkpoint.get('epoch', 'N/A')
    if epoch != 'N/A':
        epoch = epoch + 1
    
    print(f"  📦 Эпоха: {epoch}")
    print(f"  📉 Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  🎯 Train Folds: {checkpoint.get('train_folds', 'N/A')}")
    print(f"  🎯 Val Folds: {checkpoint.get('val_folds', 'N/A')}")
    print()

if __name__ == "__main__":
    print("📊 Анализ обучения UNETR\n")
    
    # Ищем все checkpoint'ы
    checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))
    
    if not checkpoints:
        print("❌ Чекпоинты не найдены в папке 'checkpoints'")
    else:
        print(f"✅ Найдено чекпоинтов: {len(checkpoints)}\n")
        
        # Анализируем каждый
        for cp in checkpoints:
            analyze_checkpoint(cp)
        
        # Если есть логи с потерями (нужно сохранить их отдельно)
        # Пример: если у тебя есть файл training_log.npy с [train_losses, val_losses]
        log_file = CHECKPOINT_DIR / "training_log.npy"
        if log_file.exists():
            losses = np.load(log_file, allow_pickle=True).item()
            plot_loss_curves(losses['train'], losses['val'])
        else:
            print("💡 Подсказка: Чтобы построить график, сохраняй потери в каждом epoch:")
            print("   np.save('checkpoints/training_log.npy', {'train': train_losses, 'val': val_losses})")
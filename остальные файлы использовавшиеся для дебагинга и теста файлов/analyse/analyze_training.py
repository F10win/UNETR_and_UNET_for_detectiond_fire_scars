import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# === НАСТРОЙКИ ===
CHECKPOINT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\checkpoints")
LOGS_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\logs\11ch")  # ← НОВАЯ ПАПКА
MODEL_FILTER = None

def find_latest_log(search_dir, keyword=None):
    """Ищет самый свежий лог в папке и подпапках"""
    if not search_dir.exists():
        return None
    
    # Ищем во всех подпапках
    logs = list(search_dir.rglob("*.txt"))
    if not logs:
        return None
    
    if keyword:
        logs = [f for f in logs if keyword in str(f)]
        
    if not logs:
        return None

    latest = max(logs, key=os.path.getmtime)
    return latest

def parse_training_log(log_path):
    if not log_path or not log_path.exists():
        return None
    
    print(f"📄 Чтение лога: {log_path}")
    
    train_losses = []
    val_losses = []
    epochs = []
    
    pattern = re.compile(r'Epoch\s+(\d+).*Train:\s+([\d.]+).*Val:\s+([\d.]+)')
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    val_losses.append(float(match.group(3)))
    except Exception as e:
        print(f"⚠️  Ошибка чтения: {e}")
        return None

    if epochs:
        return {'epochs': epochs, 'train': train_losses, 'val': val_losses}
    return None

def analyze_checkpoints(checkpoint_dir, model_filter=None):
    print(f"\n🔍 Сканирование чекпоинтов...")
    checkpoints = []
    
    for pth_file in checkpoint_dir.glob("*.pth"):
        if model_filter and model_filter not in pth_file.name:
            continue
        
        try:
            checkpoint = torch.load(pth_file, map_location='cpu', weights_only=True)
            info = {
                'file': pth_file.name,
                'size_mb': pth_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(pth_file.stat().st_mtime),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'fold': checkpoint.get('fold', 'N/A'),
                'val_loss': checkpoint.get('val_loss', float('inf')),
            }
            checkpoints.append(info)
        except Exception:
            pass
            
    checkpoints.sort(key=lambda x: x['modified'])
    return checkpoints

def main():
    print("🔬 Анализ результатов обучения")
    print("="*60)
    
    # 1. Ищем самый свежий лог
    print(f"\n🔍 Поиск логов в: {LOGS_DIR}")
    latest_log = find_latest_log(LOGS_DIR, "3ch")  # Ищи "3ch" или "11ch"
    
    if latest_log:
        print(f"✅ Найден свежий лог: {latest_log.name}")
        print(f"📁 Путь: {latest_log}")
        log_data = parse_training_log(latest_log)
    else:
        print("⚠️  Лог файлы не найдены.")
        log_data = None

    # 2. Анализируем чекпоинты
    checkpoints = analyze_checkpoints(CHECKPOINT_DIR)
    
    # 3. Строим графики
    if log_data:
        plt.figure(figsize=(10, 6))
        plt.plot(log_data['epochs'], log_data['train'], 'b-o', label='Train Loss', markersize=3, alpha=0.7)
        plt.plot(log_data['epochs'], log_data['val'], 'r-s', label='Val Loss', markersize=3, alpha=0.7)
        
        best_idx = np.argmin(log_data['val'])
        plt.axvline(x=log_data['epochs'][best_idx], color='green', linestyle='--', 
                   label=f'Best: {log_data["val"][best_idx]:.4f}')
        
        plt.title(f'Кривая обучения\n{latest_log.name}', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Dice Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(CHECKPOINT_DIR / 'learning_curve_latest.png', dpi=150)
        print("\n✅ График сохранён: checkpoints/learning_curve_latest.png")
        plt.close()

    # 4. Сводка по чекпоинтам
    if checkpoints:
        print("\n📋 Сводка по моделям (.pth):")
        print(f"{'Файл':<40} {'Fold':<6} {'Epoch':<7} {'Val Loss':<10} {'Size':<8}")
        print("-"*75)
        for cp in checkpoints:
            print(f"{cp['file']:<40} {cp['fold']:<6} {str(cp['epoch']):<7} {cp['val_loss']:<10.4f} {cp['size_mb']:<8.1f}MB")

if __name__ == "__main__":
    main()
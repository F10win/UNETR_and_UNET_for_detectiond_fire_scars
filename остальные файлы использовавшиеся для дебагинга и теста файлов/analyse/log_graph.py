import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def parse_training_log(log_file_path):
    """Парсинг лог-файла обучения"""
    folds_data = defaultdict(lambda: {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []})
    
    # Проверка наличия файла
    if not os.path.exists(log_file_path):
        print(f"❌ Ошибка: Файл не найден по пути: {log_file_path}")
        print("💡 Проверьте, правильно ли указан путь (нет ли лишних повторений папок).")
        return folds_data

    print(f"📂 Чтение файла: {log_file_path}")
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Регулярное выражение для поиска строк с метриками
    pattern = r'Fold (\d+) \| Epoch (\d+) \| Train: ([\d.]+) \| Val: ([\d.]+) \| LR: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        fold, epoch, train_loss, val_loss, lr = match
        fold = int(fold)
        epoch = int(epoch)
        train_loss = float(train_loss)
        val_loss = float(val_loss)
        lr = float(lr)
        
        folds_data[fold]['epoch'].append(epoch)
        folds_data[fold]['train_loss'].append(train_loss)
        folds_data[fold]['val_loss'].append(val_loss)
        folds_data[fold]['lr'].append(lr)
    
    return folds_data

def plot_training_curves(folds_data, save_path='training_curves.png'):
    """Визуализация кривых обучения"""
    if not folds_data:
        print("❌ Нет данных для построения графиков.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics by Fold (UNETR 11ch)', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # График 1: Training Loss
    ax1 = axes[0, 0]
    for fold_idx, (fold_num, data) in enumerate(sorted(folds_data.items())):
        color = colors[fold_idx % len(colors)]
        ax1.plot(data['epoch'], data['train_loss'], 'o-', 
                label=f'Fold {fold_num}', color=color, linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Validation Loss
    ax2 = axes[0, 1]
    for fold_idx, (fold_num, data) in enumerate(sorted(folds_data.items())):
        color = colors[fold_idx % len(colors)]
        ax2.plot(data['epoch'], data['val_loss'], 's-', 
                label=f'Fold {fold_num}', color=color, linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # График 3: Learning Rate
    ax3 = axes[1, 0]
    for fold_idx, (fold_num, data) in enumerate(sorted(folds_data.items())):
        color = colors[fold_idx % len(colors)]
        ax3.plot(data['epoch'], data['lr'], '^-', 
                label=f'Fold {fold_num}', color=color, linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # График 4: Train vs Val Loss (среднее по фолдам)
    ax4 = axes[1, 1]
    all_train = []
    all_val = []
    epochs = None
    
    # Выравнивание длин массивов (если фолды остановились на разных эпохах)
    max_len = max(len(data['epoch']) for data in folds_data.values())
    
    for fold_num, data in folds_data.items():
        # Дополняем массивы NaN до максимальной длины для корректного усреднения
        pad_len = max_len - len(data['epoch'])
        train_padded = data['train_loss'] + [np.nan] * pad_len
        val_padded = data['val_loss'] + [np.nan] * pad_len
        
        all_train.append(train_padded)
        all_val.append(val_padded)
        if epochs is None:
            epochs = list(range(1, max_len + 1))
    
    mean_train = np.nanmean(all_train, axis=0)
    mean_val = np.nanmean(all_val, axis=0)
    
    ax4.plot(epochs, mean_train, 'o-', label='Mean Train Loss', 
            color='#FF6B6B', linewidth=2, markersize=5)
    ax4.plot(epochs, mean_val, 's-', label='Mean Val Loss', 
            color='#4ECDC4', linewidth=2, markersize=5)
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Average Loss Across Folds', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен: {save_path}")
    plt.show()

# Основной код
if __name__ == "__main__":
    # Полный путь к вашему файлу (используем r"" для экранирования слешей)
    log_file = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\logs\UNETR\11ch\training_log_2026-05-11_02-37-42.txt"
    
    folds_data = parse_training_log(log_file)
    
    if folds_data:
        print(f"📊 Найдено данных для {len(folds_data)} фолдов.")
        plot_training_curves(folds_data, save_path='training_curves_11ch.png')
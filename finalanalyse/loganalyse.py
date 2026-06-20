#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация метрик обучения для 4 моделей
BasicUNET (3ch, 11ch) и UNETR (3ch, 11ch)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np
from datetime import datetime

# Настройки стиля
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# === ПУТИ К ЛОГАМ ===
LOG_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\logs\NewTrain")

# Словарь моделей: имя модели -> путь к CSV файлу
MODELS = {
    'UNETR 3ch': LOG_DIR / "UNETR" / "3channels" / "metrics_2026-05-29_02-00-56.csv",
    'UNETR 11ch': LOG_DIR / "UNETR" / "11channels" / "metrics_2026-05-27_20-58-46.csv",  # Укажите актуальный файл
    'BasicUNET 3ch': LOG_DIR / "BasicUNET" / "3channels" / "metrics_basicunet_3ch_2026-05-29_22-55-48.csv",  # Укажите актуальный файл
    'BasicUNET 11ch': LOG_DIR / "BasicUNET" / "11channels" / "metrics_basicunet_11ch_2026-05-30_02-19-17.csv",  # Укажите актуальный файл
}

# Цвета для моделей
COLORS = {
    'UNETR 3ch': '#1f77b4',      # синий
    'UNETR 11ch': '#ff7f0e',     # оранжевый
    'BasicUNET 3ch': '#2ca02c',  # зеленый
    'BasicUNET 11ch': '#d62728', # красный
}

def load_metrics(model_name, csv_path):
    """Загрузка метрик из CSV файла"""
    if not csv_path.exists():
        print(f"⚠️  Файл не найден для {model_name}: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Загружено {model_name}: {len(df)} эпох")
        return df
    except Exception as e:
        print(f"❌ Ошибка загрузки {model_name}: {e}")
        return None

def plot_training_comparison(models_data):
    """Создание сравнительных графиков обучения"""
    
    # Фильтруем только успешные загрузки
    models_data = {k: v for k, v in models_data.items() if v is not None}
    
    if not models_data:
        print("❌ Нет данных для визуализации!")
        return
    
    # Создаем фигуру с сеткой 2x2
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # === 1. Training Loss ===
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name, df in models_data.items():
        ax1.plot(df['epoch'], df['train_loss'], 
                label=model_name, 
                color=COLORS[model_name],
                linewidth=2,
                alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # === 2. Validation Loss ===
    ax2 = fig.add_subplot(gs[0, 1])
    for model_name, df in models_data.items():
        ax2.plot(df['epoch'], df['val_loss'], 
                label=model_name, 
                color=COLORS[model_name],
                linewidth=2,
                alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # === 3. Dice/F1 Score ===
    ax3 = fig.add_subplot(gs[1, 0])
    for model_name, df in models_data.items():
        ax3.plot(df['epoch'], df['val_dice'], 
                label=model_name, 
                color=COLORS[model_name],
                linewidth=2,
                alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Dice Score', fontsize=11)
    ax3.set_title('Validation Dice Score Comparison', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # === 4. IoU Score ===
    ax4 = fig.add_subplot(gs[1, 1])
    for model_name, df in models_data.items():
        ax4.plot(df['epoch'], df['val_iou'], 
                label=model_name, 
                color=COLORS[model_name],
                linewidth=2,
                alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('IoU Score', fontsize=11)
    ax4.set_title('Validation IoU Score Comparison', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    # Общий заголовок
    fig.suptitle('🔥 Сравнение архитектур UNETR и BasicUNET\n(3 канала vs 11 каналов)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Сохранение
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = LOG_DIR.parent / "analysis" / f"training_comparison_{timestamp}.png"
    output_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Графики сохранены: {output_path}")
    plt.show()
    
    return output_path

def plot_learning_curves_detailed(models_data):
    """Детальные кривые обучения с метриками"""
    
    models_data = {k: v for k, v in models_data.items() if v is not None}
    
    if not models_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('train_loss', 'Training Loss', axes[0]),
        ('val_loss', 'Validation Loss', axes[1]),
        ('val_iou', 'IoU Score', axes[2]),
        ('lr', 'Learning Rate', axes[3]),
    ]
    
    for metric, title, ax in metrics_to_plot:
        for model_name, df in models_data.items():
            ax.plot(df['epoch'], df[metric], 
                   label=model_name, 
                   color=COLORS[model_name],
                   linewidth=2,
                   alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        if metric == 'lr':
            ax.set_yscale('log')
    
    fig.suptitle('📊 Детальные метрики обучения', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = LOG_DIR.parent / "analysis" / f"detailed_metrics_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Детальные метрики: {output_path}")
    plt.show()

def print_summary_table(models_data):
    """Печать сводной таблицы лучших результатов"""
    
    print("\n" + "="*100)
    print("📋 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*100)
    print(f"{'Модель':<15} | {'Лучший Val Loss':<18} | {'Лучший Dice':<15} | {'Лучший IoU':<15} | {'Эпоха':<8}")
    print("-"*100)
    
    for model_name, df in models_data.items():
        if df is not None:
            best_idx = df['val_loss'].idxmin()
            best_row = df.loc[best_idx]
            
            print(f"{model_name:<15} | {best_row['val_loss']:<18.6f} | "
                  f"{best_row['val_dice']:<15.6f} | {best_row['val_iou']:<15.6f} | {int(best_row['epoch']):<8}")
    
    print("="*100)

def main():
    print("="*80)
    print("📊 ВИЗУАЛИЗАЦИЯ МЕТРИК ОБУЧЕНИЯ")
    print("="*80)
    
    # Загрузка данных
    models_data = {}
    for model_name, csv_path in MODELS.items():
        models_data[model_name] = load_metrics(model_name, csv_path)
    
    # Создание графиков
    print("\n📈 Генерация сравнительных графиков...")
    plot_training_comparison(models_data)
    
    print("\n📈 Генерация детальных кривых обучения...")
    plot_learning_curves_detailed(models_data)
    
    # Сводная таблица
    print_summary_table(models_data)
    
    print("\n✅ Визуализация завершена!")
    print("="*80)

if __name__ == "__main__":
    main()
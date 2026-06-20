import matplotlib.pyplot as plt
import numpy as np

# Данные из лога
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
best_epochs = [59, 57, 63, 100, 1]  # лучшие эпохи
val_losses = [0.2387, 0.0449, 0.0367, 0.0616, 0.3931]  # минимальные Val Loss

# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(10, 6))

# Создаем столбчатую диаграмму
bars = ax.bar(folds, val_losses, color=['skyblue', 'skyblue', 'red', 'skyblue', 'skyblue'], 
              edgecolor='black', linewidth=1.2, alpha=0.8)

# Добавляем значения на столбцы
for i, (bar, epoch, loss) in enumerate(zip(bars, best_epochs, val_losses)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'Epoch: {epoch}\nVal Loss: {loss:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Выделяем Fold 3 как наилучший
ax.patches[2].set_color('#FF6B6B')  # ярко-красный цвет
ax.patches[2].set_edgecolor('#CC0000')
ax.patches[2].set_linewidth(2)

# Настраиваем оси и заголовок
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('UNETR Training Results - Best Validation Loss by Fold\n(Fold 3 - Best Performance)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(val_losses) * 1.3)  # добавляем место для текста

# Добавляем сетку
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Добавляем легенду
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#FF6B6B', edgecolor='#CC0000', linewidth=2, label='Fold 3 (Best)'),
                   Patch(facecolor='skyblue', edgecolor='black', linewidth=1.2, label='Other Folds')]
ax.legend(handles=legend_elements, loc='upper right')

# Улучшаем layout
plt.tight_layout()

# Сохраняем и показываем
plt.savefig('unetr_folds_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Выводим статистику
print("=" * 60)
print("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ UNETR ПО ФОЛДАМ")
print("=" * 60)
for i, (fold, epoch, loss) in enumerate(zip(folds, best_epochs, val_losses)):
    marker = "🏆" if i == 2 else ""
    print(f"{marker} {fold}: Best Epoch = {epoch}, Val Loss = {loss:.4f}")
print("=" * 60)
print(f"✅ Наилучший результат: Fold 3 с Val Loss = 0.0367 (Epoch 63)")
print("=" * 60)
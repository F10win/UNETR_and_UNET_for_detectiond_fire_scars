import os
import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# === НАСТРОЙКИ ===
# Достаточно указать одну папку, так как маски одинаковые
DATASET_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\validation_3ch")
OUTPUT_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\analysis\NewTest\val")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_category(pct):
    """Классификация патча по проценту покрытия пожаром"""
    if pct == 0:
        return "❌ Нет гари (0%)"
    elif pct == 100:
        return "🔥 Чистая гарь (100%)"
    elif pct < 1:
        return " Следы (<1%)"
    elif pct < 20:
        return " Слабо смешанный (1-20%)"
    elif pct < 50:
        return "🟡 Средне смешанный (20-50%)"
    elif pct < 90:
        return "🟠 Сильно смешанный (50-90%)"
    else:
        return "🟥 Почти чистая гарь (90-99%)"

def analyze_masks(dataset_path):
    print(f" Анализ масок в: {dataset_path}")
    print("="*60)
    
    # Находим все маски
    mask_files = sorted(list(dataset_path.rglob("mask.tif")))
    if not mask_files:
        print(" Маски не найдены! Проверьте путь.")
        return
    
    print(f" Найдено патчей: {len(mask_files)}\n")
    
    data = []
    
    for mask_path in tqdm(mask_files, desc="Обработка патчей"):
        try:
            mask = tifffile.imread(mask_path)
            # Бинаризация: всё что > 0 считаем гарью
            binary = (mask > 0).astype(np.float32)
            fire_pct = (binary.sum() / binary.size) * 100
            
            patch_name = mask_path.parent.name
            category = get_category(fire_pct)
            
            data.append({
                "Patch": patch_name,
                "Fire_%": round(fire_pct, 2),
                "Category": category
            })
        except Exception as e:
            print(f"️ Ошибка чтения {mask_path.name}: {e}")
            
    df = pd.DataFrame(data)
    
    # === СТАТИСТИКА ===
    print("\n📊 РАСПРЕДЕЛЕНИЕ ПАТЧЕЙ ПО КАТЕГОРИЯМ:")
    print("="*60)
    category_counts = df["Category"].value_counts().sort_index()
    for cat, count in category_counts.items():
        pct_of_total = (count / len(df)) * 100
        print(f"{cat:<25} | {count:>4} патчей | {pct_of_total:>5.1f}%")
        
    print("\n📈 ОБЩАЯ СТАТИСТИКА ПОКРЫТИЯ:")
    print(f"   Средний % гари: {df['Fire_%'].mean():.2f}%")
    print(f"   Медиана:        {df['Fire_%'].median():.2f}%")
    print(f"   Std отклонение: {df['Fire_%'].std():.2f}%")
    print(f"   Мин: {df['Fire_%'].min():.2f}% | Макс: {df['Fire_%'].max():.2f}%")
    
    # === ВИЗУАЛИЗАЦИЯ ===
    plt.figure(figsize=(12, 5))
    
    # Гистограмма распределения
    plt.subplot(1, 2, 1)
    plt.hist(df["Fire_%"], bins=50, color="#2ecc71", edgecolor="black", alpha=0.8)
    plt.axvline(df["Fire_%"].mean(), color="red", linestyle="--", linewidth=2, label=f'Среднее: {df["Fire_%"].mean():.1f}%')
    plt.title("Распределение процента гари по патчам")
    plt.xlabel("Процент покрытия пожаром (%)")
    plt.ylabel("Количество патчей")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    
    # Круговая диаграмма категорий
    plt.subplot(1, 2, 2)
    counts = df["Category"].value_counts()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140, 
            colors=plt.cm.tab20(np.linspace(0, 1, len(counts))))
    plt.title("Доля категорий патчей")
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "mask_distribution_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ График сохранён: {save_path}")
    plt.close()
    
    # Сохранение в CSV
    csv_path = OUTPUT_DIR / "mask_analysis_details.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Таблица сохранена: {csv_path}")
    print("="*60)

if __name__ == "__main__":
    analyze_masks(DATASET_PATH)
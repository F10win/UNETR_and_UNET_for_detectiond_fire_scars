import pandas as pd
from pathlib import Path

print("🔍 Диагностика CSV файла\n")

# Путь к CSV
CSV_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\satellite_data.csv")

# Читаем CSV
df = pd.read_csv(CSV_PATH, sep=';')

print(f"✅ Всего записей: {len(df)}")
print(f"\n📋 Названия колонок:")
for col in df.columns:
    print(f"  - '{col}'")

print(f"\n🎨 Уникальные значения в колонке 'fold':")
if 'fold' in df.columns:
    unique_folds = df['fold'].unique()
    print(f"  Количество уникальных fold'ов: {len(unique_folds)}")
    for fold in unique_folds:
        count = (df['fold'] == fold).sum()
        print(f"  - '{fold}': {count} записей")
else:
    print("  ❌ Колонка 'fold' не найдена!")

print(f"\n📁 Примеры значений в колонке 'folder':")
if 'folder' in df.columns:
    print(df['folder'].head(10).tolist())
else:
    print("  ❌ Колонка 'folder' не найдена!")

print(f"\n📊 Первые 5 строк CSV:")
print(df.head())

# Проверка: есть ли 'purple' в fold
if 'fold' in df.columns:
    purple_count = (df['fold'] == 'purple').sum()
    print(f"\n🔍 Записей с fold='purple': {purple_count}")
    
    # Покажем какие fold'ы есть
    print(f"\n💡 Доступные fold'ы для тестирования:")
    for fold in df['fold'].unique()[:3]:  # Первые 3
        count = (df['fold'] == fold).sum()
        print(f"  - '{fold}': {count} записей")
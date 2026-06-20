#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование модели BasicUNet на новых снимках (3 канала)
Адаптировано для Sentinel-2 с 3 спектральными каналами
"""

import torch
import numpy as np
import rasterio
from rasterio.transform import xy, rowcol
from rasterio import features as rio_features
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from monai.networks.nets import BasicUNet  # ✅ Заменено на BasicUNet
from scipy import ndimage
import json
from datetime import datetime
import warnings
import xml.etree.ElementTree as ET
from shapely.geometry import shape as shapely_shape
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

print("="*80)
print("🔥 ИНФЕРЕНЦИЯ МОДЕЛИ BasicUNet (3 КАНАЛА) НА НОВЫХ СНИМКАХ")
print("="*80)

# === НАСТРОЙКИ ===
# 📁 Путь к чекпоинту BasicUNet 3ch
MODEL_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\NewTrain\basicunet_3ch_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_IMAGE = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\raw_data\region_15\20210807T020701_20210807T020656_T54VXQ_3chan_after.tif")

THRESHOLD = 0.5
MIN_BURN_AREA_PX = 100

print(f"\n📂 Модель: {MODEL_PATH}")
print(f"🖼️  Снимок: {INPUT_IMAGE}")
print(f"🔧 Device: {DEVICE}")
print(f"📡 Каналы: 3 (B4, B8, B11 - Red, NIR, SWIR)")

if torch.cuda.is_available():
    print(f"🎮 GPU обнаружена: {torch.cuda.get_device_name(0)}")
    print(f"💾 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  GPU не доступна, используется CPU")

if not INPUT_IMAGE.exists():
    print(f"\n❌ Снимок не найден: {INPUT_IMAGE}")
    input("Нажмите Enter для выхода...")
    exit()

if not MODEL_PATH.exists():
    print(f"\n❌ Модель не найдена: {MODEL_PATH}")
    input("Нажмите Enter для выхода...")
    exit()

# === ЗАГРУЗКА МОДЕЛИ ===
print("\n📥 Загрузка модели BasicUNet...")
# ✅ Инициализация BasicUNet вместо UNETR
model = BasicUNet(
    spatial_dims=2,
    in_channels=3,  # ✅ 3 канала
    out_channels=1,
    features=(32, 32, 64, 64, 128, 128),  # Стандартная конфигурация для 512×512
    act="leakyrelu",
    norm="instance",
    dropout=0.0
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Модель загружена и готова к работе")

# === ЗАГРУЗКА СНИМКА ===
print("\n📷 Загрузка снимка...")
try:
    with rasterio.open(str(INPUT_IMAGE)) as src:
        src_profile = src.profile
        src_transform = src.transform
        src_crs = src.crs
        image = src.read().astype(np.float32)
        
        print(f"📐 Размер: {src.width}×{src.height} пикселей")
        print(f"🔢 Каналов: {src.count}")
        print(f"🌍 CRS: {src.crs}")

except Exception as e:
    print(f"❌ Ошибка загрузки снимка: {e}")
    input("Нажмите Enter для выхода...")
    exit()

if image.shape[0] != 3:
    print(f"⚠️  Предупреждение: Ожидается 3 канала, найдено {image.shape[0]}.")
    print("   Используемые каналы: B4 (Red), B8 (NIR), B11 (SWIR)")

# Нормализация
print("⚙️  Нормализация данных...")
image_norm = np.clip(image / 4000.0, 0, 1)

H, W = image_norm.shape[1], image_norm.shape[2]

# === ИНФЕРЕНЦИЯ ===
print("\n🔄 Инференция на GPU...")
PATCH_SIZE = 512
OVERLAP = 128
stride = PATCH_SIZE - OVERLAP

pred_map = np.zeros((H, W), dtype=np.float32)
count_map = np.zeros((H, W), dtype=np.float32)

total_patches = ((H - PATCH_SIZE) // stride + 1) * ((W - PATCH_SIZE) // stride + 1)
print(f"   Всего патчей для обработки: {total_patches}")

processed = 0
with torch.no_grad():
    for y in range(0, H - PATCH_SIZE + 1, stride):
        for x in range(0, W - PATCH_SIZE + 1, stride):
            patch = image_norm[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
            
            output = model(patch_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            
            pred_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
            
            processed += 1
            if processed % 100 == 0:
                print(f"   Обработано: {processed}/{total_patches} ({processed/total_patches*100:.1f}%)")

print("📊 Усреднение результатов...")
pred_map = pred_map / np.maximum(count_map, 1)
print("✅ Инференция завершена!")

# === БИНАРИЗАЦИЯ ===
print("🔧 Морфологическая обработка...")
binary_mask = (pred_map > THRESHOLD).astype(np.uint8)

binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3, 3)))
binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((5, 5)))

# === ОПРЕДЕЛЕНИЕ РАЗМЕРА ПИКСЕЛЯ ===
pixel_size_m = abs(src_transform.a)
print(f"📏 Размер пикселя: {pixel_size_m}м")

# === ВЕКТОРИЗАЦИЯ ===
print("\n🔍 Векторизация контуров масок...")
burn_patches = []
geojson_features = []
current_id = 1

try:
    shapes_gen = rio_features.shapes(binary_mask.astype(np.uint8), transform=src_transform)
    
    for geom, val in shapes_gen:
        if val == 0:
            continue
        
        try:
            poly = shapely_shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue

        area_m2 = poly.area
        area_ha = area_m2 / 10000.0
        area_px = area_m2 / (pixel_size_m ** 2)

        if area_px >= MIN_BURN_AREA_PX:
            centroid = poly.centroid
            lon, lat = centroid.x, centroid.y
            row_idx, col_idx = rowcol(src_transform, lon, lat)

            burn_patches.append({
                'ID': current_id,
                'Площадь (пиксели)': int(area_px),
                'Площадь (м²)': int(area_m2),
                'Площадь (га)': round(area_ha, 4),
                'Центр X (пикс)': round(col_idx, 2),
                'Центр Y (пикс)': round(row_idx, 2),
                'Долгота (Центр)': round(lon, 6),
                'Широта (Центр)': round(lat, 6)
            })

            geojson_features.append({
                "type": "Feature",
                "geometry": poly.__geo_interface__,
                "properties": {
                    "id": current_id,
                    "area_m2": int(area_m2),
                    "area_ha": round(area_ha, 4),
                    "center_lon": round(lon, 6),
                    "center_lat": round(lat, 6)
                }
            })
            current_id += 1
            
except Exception as e:
    print(f"⚠️  Ошибка векторизации: {e}")
    import traceback
    traceback.print_exc()

total_area_px = int(binary_mask.sum())
total_area_m2 = int(total_area_px * (pixel_size_m ** 2))
total_area_ha = round(total_area_m2 / 10000, 4)
total_image_area_ha = (H * W * (pixel_size_m ** 2)) / 10000
burn_percentage = (total_area_px / (H * W)) * 100

print(f"✅ Найдено пятен: {len(burn_patches)}")
print(f"📊 Общая площадь гари: {total_area_ha:.4f} га ({burn_percentage:.2f}%)")

# === СОЗДАНИЕ ПАПКИ РЕЗУЛЬТАТОВ ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# ✅ Обновлённый путь для BasicUNet 3ch результатов
output_dir = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\analysis\inference_results_basicunet_3ch") / f"result_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n💾 Сохранение результатов в: {output_dir}")

# === 1. ВИЗУАЛИЗАЦИЯ ===
print("📊 Создание визуализации...")
fig = plt.figure(figsize=(20, 15))

ax1 = fig.add_subplot(2, 3, 1)
# ✅ Для 3 каналов показываем RGB композит: каналы [2, 1, 0] (B11, B8, B4) или [0, 1, 2]
# Предполагаем порядок: B4, B8, B11 → для натурального цвета используем [2, 1, 0] для False Color
ax1.imshow(np.transpose(image_norm[[2, 1, 0], :, :], (1, 2, 0)))  # False Color: SWIR, NIR, Red
ax1.set_title('Псевдоцвета (B11, B8, B4)', fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
im = ax2.imshow(pred_map, cmap='hot', vmin=0, vmax=1)
ax2.set_title('Предсказание (вероятности)', fontsize=14, fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(binary_mask, cmap='gray')
ax3.set_title(f'Бинарная маска\nПлощадь: {total_area_ha:.4f} га ({burn_percentage:.2f}%)', 
             fontsize=14, fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(np.transpose(image_norm[[2, 1, 0], :, :], (1, 2, 0)))
for p in burn_patches:
    ax4.plot(p['Центр X (пикс)'], p['Центр Y (пикс)'], 'r+', markersize=12, markeredgewidth=2)
    ax4.annotate(f"#{p['ID']}\n{p['Площадь (га)']:.2f} га", 
                (p['Центр X (пикс)'], p['Центр Y (пикс)']), textcoords="offset points", xytext=(10, 10),
                fontsize=8, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
ax4.set_title(f'Обнаруженные пятна ({len(burn_patches)} шт.)', fontsize=14, fontweight='bold')
ax4.axis('off')

ax5 = fig.add_subplot(2, 3, 5)
if burn_patches:
    areas = [p['Площадь (га)'] for p in burn_patches]
    ax5.hist(areas, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax5.axvline(np.mean(areas), color='blue', linestyle='--', linewidth=2, label=f'Среднее: {np.mean(areas):.2f} га')
    ax5.set_xlabel('Площадь пятна (га)', fontsize=12)
    ax5.set_ylabel('Количество', fontsize=12)
    ax5.set_title('Распределение площадей', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'Пятен не найдено', ha='center', va='center', fontsize=16)
    ax5.axis('off')

ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
# ✅ Обновлённый текст статистики для BasicUNet 3ch
stats_text = f"""
📊 СТАТИСТИКА АНАЛИЗА (BasicUNet 3 КАНАЛА)
📅 Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}
🖼️  Снимок: {INPUT_IMAGE.name}
📐 Размер: {W}×{H} px | Общая пл.: {total_image_area_ha:.2f} га
🔥 Гарь: {total_area_ha:.4f} га ({burn_percentage:.2f}%)
📏 Пятен: {len(burn_patches)} | Мин: {min([p['Площадь (га)'] for p in burn_patches]) if burn_patches else 0:.4f} га
🎯 Порог: {THRESHOLD} | Мин. площадь: {MIN_BURN_AREA_PX} px
📡 Каналы: B4, B8, B11
🤖 Модель: BasicUNet
"""
ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle(f'🔥 Анализ гарей (BasicUNet 3ch): {INPUT_IMAGE.name}', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / "analysis_visualization.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Визуализация: analysis_visualization.png")

# === 2. СОХРАНЕНИЕ РАСТРОВОГО СЛОЯ ===
print("💾 Сохранение растрового слоя...")
out_profile = src_profile.copy()
out_profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=None)
with rasterio.open(str(output_dir / "burn_mask.tif"), 'w', **out_profile) as dst:
    dst.write(binary_mask, 1)
print("✅ Растр: burn_mask.tif")

# === 3. СОХРАНЕНИЕ ВЕКТОРНОГО СЛОЯ ===
print("💾 Сохранение векторного слоя...")
geojson_data = {
    "type": "FeatureCollection",
    "features": geojson_features
}
if src_crs:
    geojson_data["crs"] = {"type": "name", "properties": {"name": src_crs.to_string()}}

with open(output_dir / "burn_polygons.geojson", 'w', encoding='utf-8') as f:
    json.dump(geojson_data, f, indent=2, ensure_ascii=False)
print("✅ Вектор: burn_polygons.geojson")

# === 4. СОЗДАНИЕ ОТЧЁТОВ ===
print("💾 Создание отчётов...")
df = pd.DataFrame(burn_patches)

if not df.empty:
    # CSV
    csv_path = output_dir / "burn_report.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("✅ Отчёт CSV: burn_report.csv")

    # XLSX
    try:
        xlsx_path = output_dir / "burn_report.xlsx"
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        print("✅ Отчёт XLSX: burn_report.xlsx")
    except Exception as e:
        print(f"⚠️  Ошибка XLSX: {e}")

    # XML
    try:
        xml_path = output_dir / "burn_report.xml"
        root = ET.Element("BurnAreaReport")
        root.set("date", datetime.now().strftime('%Y-%m-%d'))
        root.set("model", "BasicUNet")  # ✅ Добавлено указание модели
        root.set("channels", "3")  # ✅ Указано количество каналов
        root.set("image", INPUT_IMAGE.name)
        for _, row in df.iterrows():
            patch_elem = ET.SubElement(root, "Patch", ID=str(row['ID']))
            for col in df.columns:
                child = ET.SubElement(patch_elem, col.replace(' ', '_'))
                child.text = str(row[col])
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print("✅ Отчёт XML: burn_report.xml")
    except Exception as e:
        print(f"⚠️  Ошибка XML: {e}")

    # TXT
    txt_path = output_dir / "report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ОТЧЁТ: АНАЛИЗ ЛЕСНЫХ ГАРЕЙ (BasicUNet 3 КАНАЛА)\n")  # ✅ Обновлён заголовок
        f.write("="*80 + "\n\n")
        f.write(f"📅 Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
        f.write(f"🖼️  Исходный снимок: {INPUT_IMAGE.name}\n")
        f.write(f"📐 Разрешение снимка: {W}×{H} пикселей\n")
        f.write(f"🌍 Система координат: {src_crs}\n")
        f.write(f"📡 Использовано каналов: 3 (B4, B8, B11)\n")  # ✅ Обновлённое описание каналов
        f.write(f"🤖 Модель: BasicUNet\n")  # ✅ Добавлено указание модели
        f.write(f"📏 Общая площадь участка: {total_image_area_ha:.2f} га\n\n")
        f.write("-"*80 + "\n")
        f.write("🔥 РЕЗУЛЬТАТЫ СЕГМЕНТАЦИИ\n")
        f.write("-"*80 + "\n\n")
        f.write(f"✅ Общая площадь выгоревших участков: {total_area_ha:.4f} га ({burn_percentage:.2f}%)\n")
        f.write(f"🔢 Количество обнаруженных пятен: {len(df)}\n")
        f.write(f"📉 Минимальная площадь пятна (фильтр): {MIN_BURN_AREA_PX} px\n")
        f.write(f"🎯 Порог классификации: {THRESHOLD}\n\n")
        f.write("📋 ДЕТАЛИЗАЦИЯ ПО ПЯТНАМ:\n")
        f.write("-"*60 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        f.write("Генерация отчёта завершена успешно.\n")
    print("✅ Отчёт TXT: report.txt")
else:
    print("⚠️  Пятна не найдены, таблицы не созданы.")

print("\n" + "="*80)
print("🎉 АНАЛИЗ ЗАВЕРШЁН!")
print("="*80)
print(f"📁 Все результаты сохранены в: {output_dir}")
print("📄 Файлы:")
print("   1. analysis_visualization.png")
print("   2. burn_mask.tif")
print("   3. burn_polygons.geojson")
print("   4. burn_report.csv/xlsx/xml/txt")
print("="*80)

input("\nНажмите Enter для выхода...")
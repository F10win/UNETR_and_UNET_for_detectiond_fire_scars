from pathlib import Path
import pandas as pd
import glob
from PIL import Image
import numpy as np

BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
CSV_PATH = BASE_DIR / "satellite_data.csv"

df = pd.read_csv(CSV_PATH, sep=';')
first_folder = df.iloc[0]['folder']

print(f"📁 Первая папка: {first_folder}")

# Ищем во всех частях датасета
for part in range(1, 6):
    part_path = BASE_DIR / f"Satellite_burned_area_dataset_part{part}" / first_folder
    if part_path.exists():
        s2_files = list(part_path.glob("sentinel2_*.tiff"))
        print(f"\nPart {part}:")
        print(f"  Найдено файлов Sentinel-2: {len(s2_files)}")
        for f in s2_files[:3]:
            img = np.array(Image.open(f))
            print(f"  - {f.name}: форма {img.shape}")
        break
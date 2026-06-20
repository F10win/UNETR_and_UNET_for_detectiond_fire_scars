from pathlib import Path
import numpy as np
import tifffile
from osgeo import gdal
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def prepare_sentinel2_for_inference(safe_folder, output_path):
    """
    Подготавливает Sentinel-2 SAFE папку для инференса
    """
    safe_path = Path(safe_folder)
    
    # Ищем гранулу
    granule_path = list(safe_path.glob('GRANULE/*'))[0]
    img_data_path = granule_path / 'IMG_DATA'
    
    # Пути к каналам (10м разрешение: B02, B03, B04, B08)
    # (20м разрешение: B05, B06, B07, B8A, B11, B12)
    # (60м разрешение: B01, B09)
    
    bands_10m = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    bands_60m = ['B01', 'B09']  # Эти можно пропустить или ресайзить
    
    all_bands = []
    
    # Загружаем каналы 10м
    for band in bands_10m:
        band_file = list(img_data_path.glob(f'*_{band}_10m.jp2'))[0]
        with rasterio.open(band_file) as src:
            band_data = src.read(1).astype(np.float32)
            all_bands.append(band_data)
    
    # Загружаем каналы 20м и ресайзим до 10м
    for band in bands_20m:
        band_file = list(img_data_path.glob(f'*_{band}_20m.jp2'))[0]
        with rasterio.open(band_file) as src:
            band_data = src.read(1).astype(np.float32)
            # Ресайз 2x
            band_data_resized = np.zeros((band_data.shape[0] * 2, band_data.shape[1] * 2), dtype=np.float32)
            for i in range(2):
                for j in range(2):
                    band_data_resized[i::2, j::2] = band_data
            all_bands.append(band_data_resized[:band_data.shape[0]*2, :band_data.shape[1]*2])
    
    # Склеиваем все каналы
    stacked = np.stack(all_bands, axis=-1)  # (H, W, 10)
    
    # Сохраняем как TIFF
    tifffile.imwrite(output_path, stacked, compression='none')
    print(f"✅ Сохранено: {output_path}")
    print(f"📐 Размер: {stacked.shape}")
    
    return output_path

if __name__ == "__main__":
    safe_folder = r"E:\путь\к\папке\S2A_MSIL2A_..."
    output_path = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\Validate\prepared_sentinel2.tif"
    
    prepare_sentinel2_for_inference(safe_folder, output_path)
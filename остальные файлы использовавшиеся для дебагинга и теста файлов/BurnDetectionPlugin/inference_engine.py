import torch
import numpy as np
import rasterio
from rasterio.transform import xy, rowcol
from rasterio import features as rio_features
from pathlib import Path
from monai.networks.nets import UNETR
from scipy import ndimage
from shapely.geometry import shape as shapely_shape
import pandas as pd
import json
from datetime import datetime
from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsProject


class BurnDetectionInference:
    """Движок инференции для детекции гарей"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Загрузка модели UNETR"""
        print(f"📥 Загрузка модели из: {self.model_path}")
        
        # Определяем количество каналов из имени файла
        if '11ch' in self.model_path.name or '13ch' in self.model_path.name:
            in_channels = 11
        else:
            in_channels = 3
        
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=1,
            img_size=(512, 512),
            spatial_dims=2,
            feature_size=16,
            norm_name="instance"
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Модель загружена на {self.device}")
    
    def run_inference(self, raster_path, threshold=0.5, min_area_px=100, 
                      pixel_resolution=10, mask_layer=None, callback=None):
        """
        Запуск инференции на растре
        
        Args:
            raster_path: Путь к растровому файлу
            threshold: Порог бинаризации
            min_area_px: Минимальная площадь пятна в пикселях
            pixel_resolution: Разрешение пикселя в метрах
            mask_layer: Слой-маска (опционально)
            callback: Функция обратного вызова для прогресса
        
        Returns:
            dict с результатами
        """
        results = {
            'success': False,
            'mask_layer': None,
            'vector_layer': None,
            'stats': {},
            'files': {}
        }
        
        try:
            # Чтение растра
            print(f"📷 Чтение растра: {raster_path}")
            with rasterio.open(str(raster_path)) as src:
                src_profile = src.profile
                src_transform = src.transform
                src_crs = src.crs
                image = src.read().astype(np.float32)
                
                # Проверка количества каналов
                if '11ch' in self.model_path.name:
                    if image.shape[0] != 11:
                        raise ValueError(f"Ожидается 11 каналов, найдено {image.shape[0]}")
                else:
                    if image.shape[0] != 3:
                        raise ValueError(f"Ожидается 3 канала, найдено {image.shape[0]}")
            
            # Нормализация
            print("⚙️ Нормализация...")
            image_norm = np.clip(image / 4000.0, 0, 1)
            H, W = image_norm.shape[1], image_norm.shape[2]
            
            # Инференция (слайдинг окно)
            print("🔄 Инференция...")
            PATCH_SIZE = 512
            OVERLAP = 128
            stride = PATCH_SIZE - OVERLAP
            
            pred_map = np.zeros((H, W), dtype=np.float32)
            count_map = np.zeros((H, W), dtype=np.float32)
            
            total_patches = ((H - PATCH_SIZE) // stride + 1) * ((W - PATCH_SIZE) // stride + 1)
            processed = 0
            
            with torch.no_grad():
                for y in range(0, H - PATCH_SIZE + 1, stride):
                    for x in range(0, W - PATCH_SIZE + 1, stride):
                        patch = image_norm[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)
                        
                        output = self.model(patch_tensor)
                        pred = torch.sigmoid(output).squeeze().cpu().numpy()
                        
                        pred_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
                        count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
                        
                        processed += 1
                        if callback and processed % 50 == 0:
                            progress = (processed / total_patches) * 50  # 0-50%
                            callback(progress, f"Обработано: {processed}/{total_patches}")
            
            # Усреднение
            pred_map = pred_map / np.maximum(count_map, 1)
            
            if callback:
                callback(60, "Бинаризация...")
            
            # Бинаризация
            binary_mask = (pred_map > threshold).astype(np.uint8)
            
            # Морфология
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3, 3)))
            binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((5, 5)))
            
            if callback:
                callback(70, "Векторизация...")
            
            # Векторизация
            burn_patches = []
            geojson_features = []
            current_id = 1
            
            for geom, val in rio_features.shapes(binary_mask, transform=src_transform):
                if val == 0:
                    continue
                
                try:
                    poly = shapely_shape(geom)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                except:
                    continue
                
                area_m2 = poly.area
                area_ha = area_m2 / 10000.0
                area_px = area_m2 / (pixel_resolution ** 2)
                
                if area_px >= min_area_px:
                    centroid = poly.centroid
                    lon, lat = centroid.x, centroid.y
                    
                    burn_patches.append({
                        'ID': current_id,
                        'Площадь (пиксели)': int(area_px),
                        'Площадь (м²)': int(area_m2),
                        'Площадь (га)': round(area_ha, 4),
                        'Долгота': round(lon, 6),
                        'Широта': round(lat, 6)
                    })
                    
                    geojson_features.append({
                        "type": "Feature",
                        "geometry": poly.__geo_interface__,
                        "properties": {
                            "id": current_id,
                            "area_ha": round(area_ha, 4)
                        }
                    })
                    current_id += 1
            
            # Статистика
            total_area_px = int(binary_mask.sum())
            total_area_m2 = int(total_area_px * (pixel_resolution ** 2))
            total_area_ha = round(total_area_m2 / 10000, 4)
            total_image_area_ha = (H * W * (pixel_resolution ** 2)) / 10000
            burn_percentage = (total_area_px / (H * W)) * 100
            
            results['stats'] = {
                'total_area_ha': total_area_ha,
                'burn_percentage': burn_percentage,
                'num_patches': len(burn_patches),
                'image_size': f"{W}x{H}"
            }
            
            # Создание временных слоев QGIS
            print("📊 Создание слоев QGIS...")
            
            # Растровый слой (временный)
            raster_layer = QgsRasterLayer(str(raster_path), "Source Raster")
            
            # Создаем временный GeoTIFF для маски
            import tempfile
            temp_mask = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_mask.close()
            
            out_profile = src_profile.copy()
            out_profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
            
            with rasterio.open(temp_mask.name, 'w', **out_profile) as dst:
                dst.write(binary_mask, 1)
            
            mask_qgs_layer = QgsRasterLayer(temp_mask.name, "Burn Mask")
            results['mask_layer'] = mask_qgs_layer
            
            # Векторный слой (временный)
            import json
            temp_geojson = tempfile.NamedTemporaryFile(suffix='.geojson', delete=False, mode='w')
            geojson_data = {
                "type": "FeatureCollection",
                "features": geojson_features
            }
            if src_crs:
                geojson_data["crs"] = {"type": "name", "properties": {"name": src_crs.to_string()}}
            
            json.dump(geojson_data, temp_geojson, ensure_ascii=False)
            temp_geojson.close()
            
            vector_qgs_layer = QgsVectorLayer(temp_geojson.name, "Burn Polygons", "ogr")
            results['vector_layer'] = vector_qgs_layer
            
            results['success'] = True
            results['temp_files'] = [temp_mask.name, temp_geojson.name]
            
            if callback:
                callback(100, "Готово!")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
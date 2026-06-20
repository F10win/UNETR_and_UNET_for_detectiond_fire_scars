import geopandas as gpd
import rasterio
from pathlib import Path

region_path = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\raw_data\region_04")

# Загрузка SHP
shp_file = list(region_path.glob("*fire_boundary.shp"))[0]
gdf = gpd.read_file(shp_file)

print(f"📊 Количество полигонов: {len(gdf)}")
print(f"🌍 CRS SHP: {gdf.crs}")
print(f"📐 Границы полигонов: {gdf.total_bounds}")

# Загрузка TIFF для сравнения
tif_file = list(region_path.glob("*13chan_after.tif"))[0]
with rasterio.open(tif_file) as src:
    print(f"\n🌍 CRS TIFF: {src.crs}")
    print(f"📐 Границы TIFF: {src.bounds}")
    print(f"📏 Размер: {src.width}x{src.height}")

# Проверка: совпадают ли CRS?
if gdf.crs != src.crs:
    print(f"\n⚠️  ВНИМАНИЕ: CRS не совпадают!")
    print(f"   Перепроецирую SHP в {src.crs}...")
    gdf_reprojected = gdf.to_crs(src.crs)
    print(f"✅ Новый CRS: {gdf_reprojected.crs}")
    print(f"📐 Новые границы: {gdf_reprojected.total_bounds}")
else:
    print(f"\n✅ CRS совпадают!")
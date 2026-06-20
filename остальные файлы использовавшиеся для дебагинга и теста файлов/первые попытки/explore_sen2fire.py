import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === НАСТРОЙКИ ===
BASE_PATH = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Sen2Fire"

def find_npz_files(base_path, num_files=5):
    """Находит первые N .npz файлов"""
    base = Path(base_path)
    files = list(base.rglob("*.npz"))
    
    if not files:
        print(f"❌ .npz файлы не найдены в {base_path}")
        return []
    
    print(f"✅ Найдено {len(files)} файлов .npz")
    print(f"📂 Примеры путей:")
    for i, f in enumerate(files[:num_files]):
        print(f"   {i+1}. {f}")
    
    return files[:num_files]

def explore_npz(npz_path):
    """Исследует содержимое .npz файла"""
    print(f"\n{'='*70}")
    print(f"🔍 Исследование: {npz_path.name}")
    print(f"{'='*70}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Показываем ключи
        print(f"📁 Ключи в файле: {data.files}")
        
        # Извлекаем и показываем информацию
        info = {}
        for key in data.files:
            array = data[key]
            print(f"\n📊 '{key}':")
            print(f"   Форма: {array.shape}")
            print(f"   Тип: {array.dtype}")
            print(f"   Min: {array.min()}, Max: {array.max()}")
            print(f"   Mean: {array.mean():.2f}, Std: {array.std():.2f}")
            info[key] = array
        
        # Визуализация
        visualize_all(info, npz_path.name)
        
        # Проверяем есть ли пожар
        if 'label' in info:
            burned_pixels = (info['label'] > 0).sum()
            total_pixels = info['label'].size
            percent = (burned_pixels / total_pixels) * 100
            print(f"\n🔥 Статус маски:")
            print(f"   Пожарных пикселей: {burned_pixels} / {total_pixels} ({percent:.2f}%)")
            
            return percent > 0  # Возвращаем True если есть пожар
        
        return False
        
    except Exception as e:
        print(f"❌ Ошибка при чтении {npz_path.name}: {e}")
        return False

def visualize_all(data, filename):
    """Создаёт все визуализации"""
    
    # 1. Визуализация спектральных каналов
    if 'image' in data:
        visualize_channels(data['image'], filename)
    
    # 2. Визуализация маски
    if 'label' in data:
        visualize_mask(data['label'], filename)
    
    # 3. Aerosol (если есть)
    if 'aerosol' in data:
        visualize_aerosol(data['aerosol'], filename)

def visualize_channels(image, filename):
    """Визуализация спектральных каналов Sentinel-2"""
    
    # Определяем формат
    if image.ndim == 3 and image.shape[0] < 20:
        channels_first = True
        c, h, w = image.shape
    else:
        channels_first = False
        h, w, c = image.shape
    
    print(f"\n🎨 Визуализация {c} каналов ({h}×{w})")
    
    # Названия каналов
    band_names = [
        'B1 Coastal', 'B2 Blue', 'B3 Green', 'B4 Red',
        'B5 VEG RED', 'B6 VEG RED', 'B7 VEG RED', 'B8 NIR',
        'B8A NIR', 'B9 Water vapor', 'B11 SWIR', 'B12 SWIR'
    ]
    
    # Создаём фигуру 3x4
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(min(c, 12)):
        if channels_first:
            channel = image[i, :, :]
        else:
            channel = image[:, :, i]
        
        # Нормализация для отображения
        ch_min, ch_max = channel.min(), channel.max()
        if ch_max > ch_min:
            channel_norm = (channel - ch_min) / (ch_max - ch_min)
        else:
            channel_norm = channel
        
        axes[i].imshow(channel_norm, cmap='gray')
        name = band_names[i] if i < len(band_names) else f'B{i+1}'
        axes[i].set_title(f"Ch {i}: {name}\n[{ch_min:.0f} - {ch_max:.0f}]", fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle(f"Sen2Fire Spectral Channels - {filename}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f"sen2fire_channels_{Path(filename).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Каналы сохранены: {save_path}")
    plt.close()
    
    # False Color RGB (NIR, Red, Green) - каналы 8, 4, 3
    if c >= 8:
        print("\n🖼️  Создание False Color RGB (NIR=8, Red=4, Green=3)...")
        
        if channels_first:
            nir = image[7, :, :]    # B8
            red = image[3, :, :]    # B4
            green = image[2, :, :]  # B3
        else:
            nir = image[:, :, 7]
            red = image[:, :, 3]
            green = image[:, :, 2]
        
        # Нормализация каждого канала
        def norm_channel(ch):
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                return (ch - ch_min) / (ch_max - ch_min)
            return ch
        
        rgb = np.stack([
            norm_channel(red),
            norm_channel(green),
            norm_channel(nir)
        ], axis=-1)
        
        rgb = np.clip(rgb, 0, 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title(f"False Color RGB (B8-NIR, B4-Red, B3-Green)\n{filename}", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        save_path = f"sen2fire_rgb_{Path(filename).stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ False Color RGB сохранён: {save_path}")
        plt.close()

def visualize_mask(mask, filename):
    """Визуализация маски пожара"""
    
    burned_pixels = (mask > 0).sum()
    total_pixels = mask.size
    percent = (burned_pixels / total_pixels) * 100
    
    print(f"\n🔥 Маска пожара:")
    print(f"   Пожарных пикселей: {burned_pixels} / {total_pixels} ({percent:.2f}%)")
    
    plt.figure(figsize=(12, 5))
    
    # Маска
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Fire Mask\nBurned: {burned_pixels} pixels ({percent:.2f}%)", 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Гистограмма
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(mask, return_counts=True)
    plt.bar(unique, counts, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Mask Value Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = f"sen2fire_mask_{Path(filename).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Маска сохранена: {save_path}")
    plt.close()

def visualize_aerosol(aerosol, filename):
    """Визуализация аэрозоля Sentinel-5P"""
    
    plt.figure(figsize=(6, 5))
    plt.imshow(aerosol, cmap='viridis')
    plt.colorbar(label='Aerosol Index')
    plt.title(f"Sentinel-5P Aerosol\n{filename}", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    save_path = f"sen2fire_aerosol_{Path(filename).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Aerosol сохранён: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print("🛰️  Sen2Fire Dataset Explorer")
    print("="*70)
    
    # Находим файлы
    npz_files = find_npz_files(BASE_PATH, num_files=10)
    
    if not npz_files:
        exit()
    
    # Исследуем файлы, пока не найдём файл с пожаром
    files_with_fire = []
    
    for i, npz_file in enumerate(npz_files):
        has_fire = explore_npz(npz_file)
        if has_fire:
            files_with_fire.append(npz_file)
            print(f"\n✅ НАЙДЕН ФАЙЛ С ПОЖАРОМ: {npz_file.name}")
            print(f"   Сохранено визуализаций для этого файла!")
            break
    
    if not files_with_fire:
        print(f"\n⚠️  В первых {len(npz_files)} файлах нет пожаров.")
        print(f"   Попробуй файлы из scene4 (test set) — там больше пожаров.")
    
    print(f"\n{'='*70}")
    print("🎉 Исследование завершено!")
    print("="*70)
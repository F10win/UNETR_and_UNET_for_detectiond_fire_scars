import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from monai.networks.nets import UNETR

# === НАСТРОЙКИ ===
CHECKPOINT_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\UNETR_Burn\UNETR_Burn\checkpoints\unetr_3ch_fold1.pth")
SAMPLE_IMAGE_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels\patch_0001\after_3ch.tif")
MASK_PATH = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\dataset\3_channels\patch_0001\mask.tif")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("🔍 АНАЛИЗ ФАЙЛА МОДЕЛИ (CHECKPOINT)")
print("="*70)

# 1. Загрузка и анализ файла
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    print(f"\n📂 Ключи в файле: {list(checkpoint.keys())}")
    print(f"📅 Эпоха сохранения: {checkpoint.get('epoch', 'N/A')}")
    print(f"📉 Val Loss на момент сохранения: {checkpoint.get('val_loss', 'N/A')}")
    
    # Анализ весов
    model_weights = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in model_weights.values())
    print(f"⚖️  Всего параметров: {total_params:,}")
    print(f"💾 Размер весов: {sum(p.numel() * p.element_size() for p in model_weights.values()) / (1024*1024):.2f} MB")
    
    print("\n✅ Файл модели корректен и содержит веса.")
    
except Exception as e:
    print(f"❌ Ошибка чтения файла: {e}")
    exit()

# 2. Инициализация модели и загрузка весов
print("\n🏗️  Инициализация архитектуры UNETR...")
model = UNETR(
    in_channels=3,
    out_channels=1,
    img_size=(512, 512),
    spatial_dims=2,
    feature_size=16,
    norm_name="instance"
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Модель загружена в память.")

# 3. Тестовый запуск на ОДНОМ патче
print("\n🧪 Тестовая инференция на одном патче...")
try:
    if SAMPLE_IMAGE_PATH.exists():
        # Загрузка данных
        image = tifffile.imread(SAMPLE_IMAGE_PATH).astype(np.float32)
        mask = tifffile.imread(MASK_PATH).astype(np.float32)
        
        # Нормализация
        image_norm = np.clip(image / 4000.0, 0, 1)
        if image_norm.ndim == 3 and image_norm.shape[0] >= 20:
            image_norm = np.transpose(image_norm, (2, 0, 1)) # (C, H, W)
        else:
            image_norm = np.transpose(image_norm, (2, 0, 1))
        
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).to(DEVICE)
        
        # Предсказание
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(image_norm, (1, 2, 0)))
        plt.title("Input Image (3 Channels)")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(prediction, cmap='hot')
        plt.title(f"Prediction (Probabilities)\nMean: {prediction.mean():.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Инференция успешна! График построен.")
    else:
        print("⚠️  Тестовый файл не найден, пропускаем визуализацию.")
        
except Exception as e:
    print(f"❌ Ошибка при тесте: {e}")
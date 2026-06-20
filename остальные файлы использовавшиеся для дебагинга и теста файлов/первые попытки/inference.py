import torch
import numpy as np
import tifffile
from pathlib import Path
import torchvision.transforms.functional as TF
from monai.networks.nets import UNETR
import matplotlib.pyplot as plt
from PIL import Image

# Настройки
MODEL_PATH = "checkpoints/unetr_best.pth" # Путь к твоей лучшей модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (512, 512)
CHANNELS = 13

def get_model():
    model = UNETR(
        in_channels=CHANNELS, out_channels=1, img_size=IMG_SIZE,
        spatial_dims=2, feature_size=16, norm_name="instance"
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_and_visualize(image_path, output_name="result.png"):
    print(f"📥 Загрузка: {image_path}")
    
    # 1. Чтение
    try:
        img = tifffile.imread(image_path).astype(np.float32)
    except:
        img = np.array(Image.open(image_path)).astype(np.float32)

    # Сохраняем оригинальный размер для визуализации
    original_shape = img.shape[:2]  # (H, W)
    print(f"📐 Оригинальный размер: {original_shape}")

    # 2. Подготовка данных (C, H, W)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    else:
        img = np.expand_dims(img, 0)
        
    if img.shape[0] < 13:
        pad = np.zeros((13 - img.shape[0], *img.shape[1:]))
        img = np.concatenate([img, pad], axis=0)
        
    img = np.clip(img, 0, 1)
    
    # 3. Ресайз для модели
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    if img.shape[1:] != IMG_SIZE:
        img_tensor = TF.resize(img_tensor, IMG_SIZE, antialias=True)
        
    # 4. Предсказание
    model = get_model()
    print("🔄 Предсказание модели...")
    with torch.no_grad():
        logits = model(img_tensor)
        mask = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    print(f"📐 Размер маски (после модели): {mask.shape}")
    
    # 5. Ресайз маски обратно к оригинальному размеру!
    if mask.shape != original_shape:
        mask_resized = TF.resize(
            torch.from_numpy(mask).unsqueeze(0),
            original_shape,
            interpolation=TF.InterpolationMode.BILINEAR
        ).squeeze(0).numpy()
        print(f"📐 Размер маски (ресайз): {mask_resized.shape}")
    else:
        mask_resized = mask
    
    # Бинаризация
    mask_binary = (mask_resized > 0.5).astype(np.float32)
    
    # 6. Визуализация
    plt.figure(figsize=(15, 5))
    
    # Оригинал (RGB каналы)
    if img.shape[0] > 3:
        rgb = np.transpose(img[[3, 2, 1], :, :], (1, 2, 0))
    else:
        rgb = img[0]
        
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("Original Image (RGB)")
    plt.axis('off')
    
    # Предсказание (ресайженная маска)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap='gray')
    plt.title(f"Prediction\nBurned: {(mask_binary > 0.5).sum() / mask_binary.size * 100:.1f}%")
    plt.axis('off')
    
    # Наложение (теперь размеры совпадают!)
    plt.subplot(1, 3, 3)
    overlay = rgb.copy()
    overlay[mask_binary > 0.5] = [1, 0, 0]  # Красный цвет для гари
    plt.imshow(overlay)
    plt.title("Overlay (Red = Fire)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=150)
    print(f"✅ Результат сохранён: {output_name}")
    plt.show()

if __name__ == "__main__":
    # Укажи путь к ЛЮБОМУ tiff файлу из датасета для теста
    test_file = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\Validate\S2A_MSIL1C_20240809T030541_N0511_R075_T51VWK_20240809T064500.tif"
    
    if Path(test_file).exists():
        predict_and_visualize(test_file, "prediction_result.png")
    else:
        print("❌ Файл не найден, укажи правильный путь в коде!")
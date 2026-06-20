import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
from monai.networks.nets import UNETR
from PIL import Image

# --- НАСТРОЙКИ ---
MODEL_PATH = "checkpoints/unet_best_fixed.pth"
TEST_IMAGE_PATH = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\Validate\2021-09-01-yakutsk.png"

IMG_SIZE = (512, 512)
IN_CHANNELS_REQUIRED = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_path, device):
    """Загрузка модели UNETR (12 каналов)"""
    print(f"🚀 Загрузка модели UNETR на устройство: {device}")
    
    model = UNETR(
        in_channels=IN_CHANNELS_REQUIRED,
        out_channels=1,
        img_size=IMG_SIZE,
        spatial_dims=2,
        feature_size=16,
        norm_name="instance"
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена.")
    return model

def prepare_rgb_for_sentinel2(image_path):
    """
    Загружает RGB картинку и превращает её в 'фейковый' Sentinel-2 (12 каналов).
    """
    print(f"📷 Загрузка RGB изображения: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    
    original_shape = img.shape[:2]  # (H, W)
    print(f"📐 Оригинальный размер: {original_shape}")
    
    # (H, W, 3) -> (3, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    
    # Ресайз до 512x512
    img_tensor = TF.resize(img_tensor, IMG_SIZE, antialias=True)
    
    # ПАДДИНГ: Добавляем нули, чтобы стало 12 каналов
    zeros = torch.zeros(9, IMG_SIZE[0], IMG_SIZE[1])
    img_12ch = torch.cat([img_tensor, zeros], dim=0)
    
    print(f"✅ RGB преобразован в 12-канальный тензор (512×512)")
    return img_12ch.unsqueeze(0), img, original_shape

def predict_and_visualize(model, input_tensor, original_rgb, original_shape):
    """Предсказание и рисование"""
    print("🔄 Выполнение предсказания...")
    
    with torch.no_grad():
        logits = model(input_tensor.to(DEVICE))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    print(f"📐 Размер маски (512×512): {probs.shape}")
    
    # Бинаризация (на размере 512×512)
    mask_512 = (probs > 0.5).astype(np.float32)
    
    # 🔥 РЕСАЙЗ МАСКИ ОБРАТНО К ОРИГИНАЛЬНОМУ РАЗМЕРУ!
    mask_tensor = torch.from_numpy(mask_512).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
    mask_resized = TF.resize(
        mask_tensor, 
        original_shape,  # (H, W) оригинала
        interpolation=TF.InterpolationMode.NEAREST
    ).squeeze(0).squeeze(0).numpy()
    
    print(f"📐 Размер маски (оригинал): {mask_resized.shape}")
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # 1. Оригинал RGB
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title(f"Original RGB Image\nShape: {original_rgb.shape}")
    plt.axis('off')
    
    # 2. Карта вероятностей (512×512)
    plt.subplot(1, 3, 2)
    plt.imshow(probs, cmap='gray')
    plt.title("Prediction Probability (512×512)")
    plt.axis('off')
    
    # 3. Оверлей (на оригинальном размере)
    rgb_copy = original_rgb.copy()
    rgb_copy[mask_resized > 0.5] = [1, 0, 0]  # Красный цвет
    
    plt.subplot(1, 3, 3)
    burned_percent = mask_resized.sum() / mask_resized.size * 100
    plt.imshow(rgb_copy)
    plt.title(f"Overlay (Red = Fire)\nPixels: {burned_percent:.2f}%")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("rgb_inference_result.png", dpi=150, bbox_inches='tight')
    print("✅ Результат сохранен: rgb_inference_result.png")
    plt.show()

if __name__ == "__main__":
    try:
        model = get_model(MODEL_PATH, DEVICE)
        input_tensor, original_rgb, original_shape = prepare_rgb_for_sentinel2(TEST_IMAGE_PATH)
        predict_and_visualize(model, input_tensor, original_rgb, original_shape)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
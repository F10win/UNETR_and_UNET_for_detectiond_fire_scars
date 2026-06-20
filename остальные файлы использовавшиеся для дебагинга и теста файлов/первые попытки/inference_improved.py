import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
from monai.networks.nets import UNETR
import os
from PIL import Image

# --- НАСТРОЙКИ ---
MODEL_PATH = "checkpoints/unetr_11ch_fold1.pth"
TEST_IMAGE_PATH = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Ilusha\raw_data\region_03\S2B_MSIL2A_20210906T025539_N0500_R032_T52VDQ_20230119T035340_13chan_after.tif"

IMG_SIZE = (512, 512)
IN_CHANNELS = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    """Инициализация и загрузка модели"""
    print(f"🚀 Загрузка модели на устройство: {DEVICE}")
    
    model = UNETR(
        in_channels=IN_CHANNELS,
        out_channels=1,
        img_size=IMG_SIZE,
        spatial_dims=2,
        feature_size=16,
        norm_name="instance"
    ).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена успешно!")
    return model

def find_mask_path(image_path):
    """Находит путь к маске в той же папке"""
    img_dir = Path(image_path).parent
    mask_files = list(img_dir.glob("*_mask.tiff")) + list(img_dir.glob("*_mask.png"))
    if mask_files:
        return str(mask_files[0])
    return None

def preprocess_image(image_path):
    """Загрузка и предобработка изображения"""
    print(f"📷 Загрузка изображения: {image_path}")
    
    img = tifffile.imread(image_path).astype(np.float32)
    
    print(f"📐 Исходная форма: {img.shape}")
    
    # Определяем формат
    if img.ndim == 3:
        if img.shape[0] < 20:  # Скорее всего (C, H, W)
            c, h, w = img.shape
            channels_first = True
        else:  # Скорее всего (H, W, C)
            h, w, c = img.shape
            channels_first = False
    else:
        h, w = img.shape
        c = 1
        channels_first = False
    
    original_shape = (h, w)
    
    # Приводим к (H, W, C)
    if channels_first:
        img_hwc = np.transpose(img, (1, 2, 0))
    else:
        img_hwc = img
    
    # Нормализация
    if img_hwc.max() > 10:
        img_hwc = img_hwc / 10000.0
    img_hwc = np.clip(img_hwc, 0, 1)
    
    # Подгонка каналов
    c = img_hwc.shape[2]
    if c < IN_CHANNELS:
        pad = np.zeros((h, w, IN_CHANNELS - c))
        img_hwc = np.concatenate([img_hwc, pad], axis=2)
    elif c > IN_CHANNELS:
        img_hwc = img_hwc[:, :, :IN_CHANNELS]
    
    # Ресайз для модели
    img_tensor = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    img_tensor_resized = TF.resize(img_tensor, IMG_SIZE, antialias=True)
    
    return img_tensor_resized, original_shape, img_hwc

def predict(model, input_tensor):
    """Запуск предсказания"""
    print("🔄 Выполнение предсказания...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
    return probs

def load_gt_mask(mask_path, original_shape):
    """Загружает и подготавливает Ground Truth маску"""
    if not mask_path:
        return None
        
    print(f"🛡️  Загрузка маски: {mask_path}")
    
    if mask_path.endswith('.tiff') or mask_path.endswith('.tif'):
        mask = tifffile.imread(mask_path).astype(np.float32)
    else:
        mask = np.array(Image.open(mask_path)).astype(np.float32)
    
    # Ресайз
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
    mask_resized = TF.resize(
        mask_tensor, 
        original_shape,
        interpolation=TF.InterpolationMode.NEAREST
    ).squeeze(0).numpy()
    
    # Бинаризация (порог 37)
    gt_binary = (mask_resized >= 37).astype(np.float32)
    return gt_binary

def visualize(original_img_hwc, mask_prob, original_shape, gt_mask=None):
    """Визуализация результатов (с NBR слева и кастомным RGB)"""
    print("🎨 Визуализация...")
    
    h, w = original_shape
    
    # 1. Расчет индекса гари NBR
    if original_img_hwc.shape[2] >= 12:
        nir = original_img_hwc[:, :, 7]
        swir = original_img_hwc[:, :, 11]
        nbr = (nir - swir) / (nir + swir + 1e-5)
    else:
        nir = original_img_hwc[:, :, 7]
        red = original_img_hwc[:, :, 3]
        nbr = (nir - red) / (nir + red + 1e-5)

    # 2. Подготовка КАСТОМНОГО RGB (Каналы 12, 8, 4 -> Индексы 11, 7, 3)
    # 12 (SWIR2) -> Red channel
    # 8  (NIR)   -> Green channel
    # 4  (Red)   -> Blue channel
    if original_img_hwc.shape[2] >= 12:
        rgb_display = original_img_hwc[:, :, [11, 7, 3]]
    elif original_img_hwc.shape[2] >= 4:
        # Fallback к обычному RGB если мало каналов
        rgb_display = original_img_hwc[:, :, [3, 2, 1]]
    else:
        rgb_display = original_img_hwc
    
    # 3. Предсказание (ресайз обратно)
    mask_tensor = torch.from_numpy(mask_prob).unsqueeze(0)
    mask_resized = TF.resize(
        mask_tensor, 
        (h, w),
        interpolation=TF.InterpolationMode.BILINEAR
    ).squeeze(0).numpy()
    mask_binary = (mask_resized > 0.5).astype(np.float32)
    
    # 4. Оверлей
    overlay = rgb_display.copy()
    overlay[mask_binary > 0.5] = [1.0, 0.0, 0.0]
    
    # Настройка фигуры
    num_plots = 5 if gt_mask is not None else 4
    fig_width = 4 * num_plots
    plt.figure(figsize=(fig_width, 5))
    
    # --- КАРТИНКА 1: ИНДЕКС ГАРЯ (NBR) ---
    plt.subplot(1, num_plots, 1)
    plt.imshow(nbr, cmap='RdYlGn') 
    plt.title("Burn Index (NBR)")
    plt.axis('off')
    
    # --- КАРТИНКА 2: КАСТОМНЫЙ RGB (12, 8, 4) ---
    plt.subplot(1, num_plots, 2)
    plt.imshow(rgb_display)
    plt.title(f"False Color RGB\nChannels: 12, 8, 4")
    plt.axis('off')
    
    # --- КАРТИНКА 3: Карта вероятностей ---
    plt.subplot(1, num_plots, 3)
    plt.imshow(mask_prob, cmap='gray')
    plt.title("Prediction Probability")
    plt.axis('off')
    
    # --- КАРТИНКА 4: Предсказание модели ---
    plt.subplot(1, num_plots, 4)
    plt.imshow(overlay)
    burned_percent = mask_binary.sum() / mask_binary.size * 100
    plt.title(f"Model Prediction\nBurned: {burned_percent:.2f}%")
    plt.axis('off')
    
    # --- КАРТИНКА 5: Ground Truth (если есть) ---
    if gt_mask is not None:
        gt_percent = gt_mask.sum() / gt_mask.size * 100
        overlay_gt = rgb_display.copy()
        overlay_gt[gt_mask > 0.5] = [1.0, 0.0, 0.0]
        
        plt.subplot(1, num_plots, 5)
        plt.imshow(overlay_gt)
        plt.title(f"Ground Truth\nBurned: {gt_percent:.2f}%")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("inference_result.png", dpi=150, bbox_inches='tight')
    print("✅ Результат сохранён: inference_result.png")
    plt.show()

if __name__ == "__main__":
    try:
        model = get_model()
        input_tensor, original_shape, original_img = preprocess_image(TEST_IMAGE_PATH)
        mask_prob = predict(model, input_tensor)
        
        mask_gt_path = find_mask_path(TEST_IMAGE_PATH)
        gt_mask = load_gt_mask(mask_gt_path, original_shape)
        
        visualize(original_img, mask_prob, original_shape, gt_mask)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
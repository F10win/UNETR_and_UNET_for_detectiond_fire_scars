import matplotlib.pyplot as plt
import numpy as np
import tifffile
from pathlib import Path

# Путь к файлу, который ты тестировал
test_tiff = r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset\Satellite_burned_area_dataset_part3\EMSR211_02SONEJA_02GRADING_MAP_v1_vector\sentinel2_2017-05-06.tiff"
test_path = Path(test_tiff)
folder = test_path.parent

# Ищем оригинальную маску в той же папке
mask_files = list(folder.glob("*_mask.tiff")) + list(folder.glob("*_mask.png"))
if not mask_files:
    print("❌ Оригинальная маска не найдена рядом с изображением!")
else:
    mask_path = mask_files[0]
    print(f"✅ Найдена оригинальная маска: {mask_path.name}")
    
    # Читаем данные для отображения
    # 1. Оригинал (RGB)
    img_data = tifffile.imread(test_tiff).astype(np.float32)
    if img_data.ndim == 3: img_data = np.transpose(img_data, (1, 2, 0)) # H,W,C
    rgb = (img_data[:, :, [3, 2, 1]] / img_data[:, :, [3, 2, 1]].max() * 255).astype(np.uint8)
    
    # 2. Оригинальная маска
    gt_mask = tifffile.imread(mask_path).astype(np.float32)
    gt_mask_binary = (gt_mask > 0).astype(np.float32) # Бинаризация как в датасете
    
    # 3. Твое предсказание (берем из сохраненного файла prediction_result.png)
    # Если файла нет, покажем просто GT
    pred_path = Path("prediction_result.png")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    if pred_path.exists():
        pred_img = plt.imread(pred_path)
        # Это RGB картинка, берем красный канал как маску
        # Или проще: покажем GT, так как pred_img это уже overlay
        plt.subplot(1, 3, 2)
        plt.imshow(pred_img) 
        plt.title("Model Prediction (Overlay)")
        plt.axis('off')
    else:
        plt.subplot(1, 3, 2)
        plt.imshow(rgb)
        plt.title("Model Prediction (Not Found)")
        plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask_binary, cmap='gray')
    plt.title(f"Ground Truth Mask\nBurned: {gt_mask_binary.sum() / gt_mask_binary.size * 100:.1f}%")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
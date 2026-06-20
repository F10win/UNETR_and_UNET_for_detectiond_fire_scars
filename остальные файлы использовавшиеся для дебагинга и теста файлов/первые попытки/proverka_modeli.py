import torch

# Загружаем чекпоинт
checkpoint = torch.load("checkpoints_unet/unet_3ch_best.pth", map_location='cpu', weights_only=False)

print("🔍 Ключи в чекпоинте:")
print("="*60)

# Если это словарь с model_state_dict
if 'model_state_dict' in checkpoint:
    print("✅ Чекпоинт содержит 'model_state_dict'")
    keys = list(checkpoint['model_state_dict'].keys())
    print(f"\n📊 Всего параметров: {len(keys)}")
    print(f"\n📋 Первые 10 ключей:")
    for key in keys[:10]:
        print(f"  - {key}")
    
    # Проверяем тип архитектуры по ключам
    if any('vit' in k for k in keys):
        print("\n✅ Это UNETR (содержит ViT блоки)")
    elif any('encoder' in k and 'decoder' in k for k in keys):
        print("\n⚠️  Это U-Net (содержит encoder/decoder)")
    else:
        print("\n❓ Неизвестная архитектура")
        
else:
    print("❌ Чекпоинт НЕ содержит 'model_state_dict'")
    print(f"\n📋 Доступные ключи: {list(checkpoint.keys())}")
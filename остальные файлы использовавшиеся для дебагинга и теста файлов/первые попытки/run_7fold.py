import pandas as pd
from pathlib import Path
import torch
from train import train # Импортируем функцию train из прошлого файла

def run_7fold():
    BASE_DIR = Path(r"E:\миигаик\1_Магистратура\Диплом\проект\UNETR_Burn\Satellite Burned Area Dataset")
    CSV_PATH = BASE_DIR / "satellite_data.csv"
    
    # Читаем все доступные фолды
    df = pd.read_csv(CSV_PATH, sep=';')
    ALL_FOLDS = sorted(df['fold'].unique().tolist())
    
    print(f"🌍 Начата 7-Fold Кросс-Валидация")
    print(f"📋 Fold'ы: {ALL_FOLDS}\n")
    
    results = []
    
    for val_fold in ALL_FOLDS:
        # Формируем списки: 1 фолд валидация, остальные 6 обучение
        train_folds = [f for f in ALL_FOLDS if f != val_fold]
        
        print(f"{'='*40}")
        print(f" ЗАПУСК ФОЛДА: {val_fold}")
        print(f"📊 Train: {train_folds}")
        print(f"📊 Val:   [{val_fold}]")
        
        model_path = f"checkpoints/unetr_best_fold_{val_fold}.pth"
        
        # Запускаем обучение
        try:
            val_loss = train(train_folds, [val_fold], result_path=model_path)
            results.append({'Fold': val_fold, 'Val Loss': val_loss})
        except Exception as e:
            print(f"❌ Ошибка в фолде {val_fold}: {e}")
            
    # Сводная таблица
    results_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(results_df)
    print(f"\n📉 Средний Val Loss: {results_df['Val Loss'].mean():.4f}")
    print(f"📉 Медиана Val Loss: {results_df['Val Loss'].median():.4f}")
    
    # Сохраняем таблицу
    results_df.to_csv("7fold_results.csv", index=False)
    print("💾 Таблица сохранена в 7fold_results.csv")

if __name__ == "__main__":
    run_7fold()
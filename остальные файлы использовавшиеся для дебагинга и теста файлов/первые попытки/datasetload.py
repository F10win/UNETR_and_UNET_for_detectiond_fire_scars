from datasets import load_dataset

ds = load_dataset("ibm-nasa-geospatial/burn_intensity")

# Сохранить на диск
ds.save_to_disk("F:\dataset")
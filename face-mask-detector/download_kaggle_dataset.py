import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# === Kaggle API kimliğini ayarla ===
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()  # .json bu klasörde

# === API'yi başlat ===
api = KaggleApi()
api.authenticate()

# === Dataset indir ===
dataset_name = "omkargurav/face-mask-dataset"
download_path = "data"

print("📥 Veri indiriliyor...")
api.dataset_download_files(dataset_name, path=download_path, unzip=True)
print("✅ İndirme ve açma tamamlandı!")

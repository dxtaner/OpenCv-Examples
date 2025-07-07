import os
import requests

def download_cascade(cascade_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print("Cascade dosyası indiriliyor...")
        r = requests.get(cascade_url)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print("Cascade indirildi.")
    else:
        print("Cascade dosyası zaten mevcut.")

# Cascade URL ve kayıt yolu
cascade_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade_path = "cascade_files/cascade.xml"

download_cascade(cascade_url, cascade_path)

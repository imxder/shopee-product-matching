import gdown
import zipfile
import os

file_id = '1xyY9cT-bdGEGp0VbvmY9G_wl4Ia_8Wbi'  
zip_filename = 'images.zip'
download_url = f'https://drive.google.com/uc?id={file_id}'

dest_folder = os.path.join('static', 'images')

os.makedirs(dest_folder, exist_ok=True)

print("Baixando ZIP...")
gdown.download(download_url, zip_filename, quiet=False)

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('static/')

os.remove(zip_filename)

print(f"Imagens extraídas com sucesso para: {dest_folder}")

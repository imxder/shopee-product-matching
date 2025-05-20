import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from sentence_transformers import SentenceTransformer

print("[1/2] Gerando embeddings de texto...")

df = pd.read_csv("train.csv")

text_model = SentenceTransformer('all-MiniLM-L6-v2')

tqdm.pandas()
text_embeddings = df['title'].progress_apply(lambda x: text_model.encode(str(x)))
text_embeddings = np.vstack(text_embeddings.values)

np.save("text_embeddings.npy", text_embeddings)
print("Embeddings de texto salvos como 'text_embeddings.npy'")

print("\n[2/2] Gerando embeddings de imagem...")

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_model = resnet50(pretrained=True)
img_model = torch.nn.Sequential(*list(img_model.children())[:-1])  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_dir = "static/images"
image_embeddings = []

for img_name in tqdm(df['image']):
    img_path = os.path.join(image_dir, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = img_model(image).squeeze().cpu().numpy()
            image_embeddings.append(features)
    except Exception as e:
        print(f"Erro ao processar {img_name}: {e}")
        image_embeddings.append(np.zeros(2048))  # padding se erro

image_embeddings = np.vstack(image_embeddings)
np.save("image_embeddings.npy", image_embeddings)
print("Embeddings de imagem salvos como 'image_embeddings.npy'")

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template, send_from_directory, redirect, url_for

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'images')
TRAIN_CSV = 'train.csv'
IMAGE_EMBEDDINGS_FILE = 'image_embeddings.npy'
GNN_EMBEDDINGS_FILE = 'gnn_embeddings.npy'
IMAGE_IDS_FILE = 'image_ids.npy'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
N_NEIGHBORS = 6 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def check_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Erro: O arquivo '{filepath}' não foi encontrado.")

check_file(TRAIN_CSV)
check_file(IMAGE_EMBEDDINGS_FILE)
check_file(GNN_EMBEDDINGS_FILE)
check_file(IMAGE_IDS_FILE)
if not os.path.exists(IMAGE_FOLDER):
     raise FileNotFoundError(f"Erro: A pasta de imagens '{IMAGE_FOLDER}' não foi encontrada.")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("Carregando modelos e dados...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

img_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
img_model = torch.nn.Sequential(*list(img_model.children())[:-1])
img_model.to(device)
img_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

all_image_embeddings = np.load(IMAGE_EMBEDDINGS_FILE)
all_gnn_embeddings = np.load(GNN_EMBEDDINGS_FILE)
image_ids = np.load(IMAGE_IDS_FILE, allow_pickle=True) 

df = pd.read_csv(TRAIN_CSV) #
df['image'] = df['image'].astype(str)
image_to_title = pd.Series(df.title.values, index=df.image).to_dict()
max_index = len(df) - 1

if len(all_image_embeddings) != len(all_gnn_embeddings) or len(all_image_embeddings) != len(image_ids):
    raise ValueError("Os arquivos de embeddings e IDs não têm o mesmo número de itens!")
print(f"Dimensão GNN Embeddings: {all_gnn_embeddings.shape}")
print(f"Dimensão Image Embeddings: {all_image_embeddings.shape}")

print("Treinando modelos KNN...")
nn_model_gnn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', algorithm='brute')
nn_model_gnn.fit(all_gnn_embeddings)

nn_model_img = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', algorithm='brute')
nn_model_img.fit(all_image_embeddings)
print("Pronto!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad():
            embedding = img_model(batch_t)
        return embedding.squeeze().cpu().numpy().reshape(1, -1)
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

def find_similar(embedding, nn_model):
    distances, indices = nn_model.kneighbors(embedding)
    results = []
    for i, idx in enumerate(indices.flatten()):
        img_id = image_ids[idx]
        results.append({
            'image': img_id,
            'title': image_to_title.get(img_id, "Título não encontrado"),
            'similaridade': 1 - distances.flatten()[i]
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def index_search():
    produto = None
    recomendados = None
    error_message = None

    if request.method == 'POST':
        try:
            produto_idx = int(request.form['produto_idx'])
            if 0 <= produto_idx <= max_index:
                selected_gnn_embedding = all_gnn_embeddings[produto_idx].reshape(1, -1)
                recomendados = find_similar(selected_gnn_embedding, nn_model_gnn)
                produto = recomendados.pop(0) 
                produto['is_upload'] = False
                produto['search_type'] = 'GNN' 
            else:
                error_message = f"Índice inválido. Por favor, insira um valor entre 0 e {max_index}."
        except ValueError:
            error_message = "Por favor, insira um número válido."
        except Exception as e:
            error_message = f"Ocorreu um erro: {e}"

    return render_template('index.html', max_index=max_index, produto=produto, recomendados=recomendados, error_message=error_message)


@app.route('/upload', methods=['POST'])
def upload_search():
    if 'query_image' not in request.files:
        return render_template('index.html', max_index=max_index, error_message="Nenhum arquivo enviado.")

    file = request.files['query_image']

    if file.filename == '':
        return render_template('index.html', max_index=max_index, error_message="Nenhum arquivo selecionado.")

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        query_img_embedding = generate_embedding(filename)

        if query_img_embedding is not None:
            recomendados = find_similar(query_img_embedding, nn_model_img)

            produto = {
                'image': file.filename,
                'title': 'Sua Imagem Carregada',
                'is_upload': True,
                'search_type': 'Visual'
            }
            if recomendados and recomendados[0]['similaridade'] > 0.999:
                 recomendados.pop(0)

            return render_template('index.html', max_index=max_index, produto=produto, recomendados=recomendados[:5])
        else:
            return render_template('index.html', max_index=max_index, error_message="Erro ao processar a imagem.")
    else:
        return render_template('index.html', max_index=max_index, error_message="Formato de arquivo não permitido.")

@app.route('/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/images/<filename>')
def serve_images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
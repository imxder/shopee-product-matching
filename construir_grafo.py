import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

print("🔄 Carregando embeddings...")
text_emb = np.load('text_embeddings.npy')
img_emb = np.load('image_embeddings.npy')

text_emb_norm = normalize(text_emb)
img_emb_norm = normalize(img_emb)

peso_texto = 0.5
peso_imagem = 0.5
combined_emb = np.hstack([peso_texto * text_emb_norm, peso_imagem * img_emb_norm])
combined_emb = torch.tensor(combined_emb, dtype=torch.float)

k = 5
knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(combined_emb)
distances, indices = knn.kneighbors(combined_emb)

edge_index = []
for i in range(indices.shape[0]):
    for j in range(1, k + 1): 
        edge_index.append([i, indices[i][j]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
data = Data(x=combined_emb, edge_index=edge_index)

torch.save(data, 'produto_grafo.pt')
print(f"Grafo salvo como 'produto_grafo.pt' com {data.num_nodes} nós e {data.num_edges} arestas.")

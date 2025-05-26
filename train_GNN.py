import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

data = torch.load("produto_grafo.pt", weights_only=False).to(device) 

class LightSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

model = LightSAGE(in_channels=data.num_node_features,
                  hidden_channels=512,
                  out_channels=256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def contrastive_loss_cosine(embeddings, edge_index, margin=0.5, batch_size=256, neg_sample_size=256):
    num_edges = edge_index.size(1)
    batch_indices = torch.randint(0, num_edges, (batch_size,), device=device)
    src = edge_index[0, batch_indices]
    dst = edge_index[1, batch_indices]

    z_src = embeddings[src]
    z_dst = embeddings[dst]

    positive_dist = 1 - F.cosine_similarity(z_src, z_dst)

    neg_indices = torch.randint(0, embeddings.size(0), (batch_size, neg_sample_size), device=device)
    z_neg = embeddings[neg_indices]

    z_src_exp = z_src.unsqueeze(1).expand(-1, neg_sample_size, -1)

    neg_sim = F.cosine_similarity(z_src_exp, z_neg, dim=-1)
    hardest_neg_sim, _ = neg_sim.max(dim=1)
    hardest_negative_dist = 1 - hardest_neg_sim

    loss = F.relu(margin + positive_dist - hardest_negative_dist)
    return loss.mean()


print("Treinando modelo LightSAGE")
epochs = 50
steps_per_epoch = 100

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}"):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        loss = contrastive_loss_cosine(embeddings, data.edge_index, batch_size=256)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / steps_per_epoch
    print(f"Epoch {epoch}/{epochs}, Loss Média: {avg_loss:.4f}")

print("\nGerando e salvando embeddings GNN finais...")
model.eval()
with torch.no_grad():
    final_embeddings = model(data.x, data.edge_index).cpu().numpy()

np.save("gnn_embeddings.npy", final_embeddings)
print(f"Embeddings GNN ({final_embeddings.shape}) salvos como 'gnn_embeddings.npy'")
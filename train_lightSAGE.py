import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np

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
        return x

model = LightSAGE(in_channels=data.num_node_features,
                  hidden_channels=512,
                  out_channels=256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def contrastive_loss_batch_approx(embeddings, edge_index, margin=1.0, batch_size=128, neg_sample_size=512):
    num_edges = edge_index.size(1)
    device = embeddings.device
    
    batch_indices = torch.randint(0, num_edges, (batch_size,), device=device)
    src = edge_index[0, batch_indices]
    dst = edge_index[1, batch_indices]
    
    z_src = embeddings[src]
    z_dst = embeddings[dst]
    
    positive = F.pairwise_distance(z_src, z_dst)
    

    neg_indices = torch.randint(0, embeddings.size(0), (batch_size, neg_sample_size), device=device)
    z_neg = embeddings[neg_indices]  # (batch_size, neg_sample_size, embedding_dim)
    
    z_src_exp = z_src.unsqueeze(1).expand(-1, neg_sample_size, -1)  # (batch_size, neg_sample_size, embedding_dim)
    negative = F.pairwise_distance(z_src_exp.reshape(-1, z_src.size(1)), 
                                   z_neg.reshape(-1, z_src.size(1))).view(batch_size, neg_sample_size)
    
    hardest_negative, _ = negative.min(dim=1)  # escolhe o negativo mais próximo
    
    loss = F.relu(margin + positive - hardest_negative)
    return loss.mean()

print("Treinando modelo LightSAGE")

for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = contrastive_loss_batch_approx(out, data.edge_index, margin=1.0, batch_size=128, neg_sample_size=512)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache() 
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")


print("Shape dos embeddings finais:", out.shape)
np.save("gnn_embeddings.npy", out.cpu().detach().numpy())

print("Embeddings 'gnn_embeddings.npy'")

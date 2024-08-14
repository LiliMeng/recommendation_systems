# gnn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from common import calculate_rmse

class GNNRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GNNRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gcn1 = GCNConv(embedding_dim, embedding_dim)
        self.gcn2 = GCNConv(embedding_dim, embedding_dim)
    
    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return x

def train_gnn(train_matrix, test_matrix, log_file, eval_interval, num_epochs=100, embedding_dim=16, learning_rate=0.01):
    num_users, num_items = train_matrix.shape
    num_nodes = num_users + num_items
    
    edge_index = []
    edge_attr = []
    for i in range(num_users):
        for j in range(num_items):
            if train_matrix[i, j] > 0:
                edge_index.append([i, num_users + j])
                edge_index.append([num_users + j, i])
                edge_attr.append(train_matrix[i, j])
                edge_attr.append(train_matrix[i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    model = GNNRecommendationModel(num_users, num_items, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    with open(log_file, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            embeddings = model(edge_index)
            user_embeds = embeddings[:num_users]
            item_embeds = embeddings[num_users:]
            predictions = torch.matmul(user_embeds, item_embeds.t())
            mask = torch.tensor(train_matrix > 0, dtype=torch.bool)
            loss = loss_fn(predictions[mask], torch.tensor(train_matrix[mask], dtype=torch.float))
            loss.backward()
            optimizer.step()

            if (epoch + 1) % eval_interval == 0:
                train_rmse = calculate_rmse(predictions.detach().numpy(), train_matrix)
                test_rmse = calculate_rmse(predictions.detach().numpy(), test_matrix)
                log.write(f"{epoch + 1},{train_rmse:.4f},{test_rmse:.4f}\n")
                log.flush()
    
    return model

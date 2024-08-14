import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

# Reused functions (same as before)
def calculate_rmse(pred, actual):
    mask = actual > 0
    return np.sqrt(np.sum((pred[mask] - actual[mask]) ** 2) / np.sum(mask))

def calculate_similarity(matrix):
    return cosine_similarity(matrix)

def predict_ratings(user_similarity, ratings_matrix):
    user_mean = np.mean(ratings_matrix, axis=1).reshape(-1, 1)
    ratings_diff = ratings_matrix - user_mean
    pred = user_mean + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred

# GNN model definition
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

# Reused functions for plotting, data loading, and data preparation (same as before)
def plot_training_log(log_file):
    epochs = []
    train_rmse = []
    test_rmse = []

    with open(log_file, "r") as log:
        for line in log:
            epoch, train, test = line.strip().split(',')
            epochs.append(int(epoch))
            train_rmse.append(float(train))
            test_rmse.append(float(test))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label="Training RMSE")
    plt.plot(epochs, test_rmse, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Evaluation RMSE over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_evaluation_curve.png")
    plt.show()

def load_movielens_data():
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(url, sep='\t', names=column_names)
    df = df.drop(columns=['timestamp'])
    return df

def prepare_data(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()

    train_matrix = np.zeros((num_users, num_items))
    for line in train_data.itertuples():
        train_matrix[line[1]-1, line[2]-1] = line[3]

    test_matrix = np.zeros((num_users, num_items))
    for line in test_data.itertuples():
        test_matrix[line[1]-1, line[2]-1] = line[3]
    
    return train_matrix, test_matrix, num_users, num_items

def main(model_type, K, steps, alpha, beta, log_filename, eval_interval, embedding_dim=16, num_epochs=100):
    df = load_movielens_data()
    train_matrix, test_matrix, num_users, num_items = prepare_data(df)
    
    if model_type == "matrix_factorization":
        predictions = train_matrix_factorization(train_matrix, test_matrix, K, steps, alpha, beta, log_filename, eval_interval)
    elif model_type == "collaborative_filtering":
        predictions = train_collaborative_filtering(train_matrix, test_matrix, log_filename, eval_interval)
    elif model_type == "gnn":
        model = train_gnn(train_matrix, test_matrix, log_filename, eval_interval, num_epochs, embedding_dim)
    else:
        raise ValueError("Invalid model type. Choose 'matrix_factorization', 'collaborative_filtering', or 'gnn'.")
    
    plot_training_log(log_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommendation System on MovieLens 100K")
    parser.add_argument('--model_type', type=str, default='matrix_factorization', choices=['matrix_factorization', 'collaborative_filtering', 'gnn'], help='Model type to use')
    parser.add_argument('--K', type=int, default=20, help='Number of latent features (only used for matrix factorization)')
    parser.add_argument('--steps', type=int, default=5000, help='Number of iterations (only used for matrix factorization)')
    parser.add_argument('--alpha', type=float, default=0.0002, help='Learning rate (only used for matrix factorization)')
    parser.add_argument('--beta', type=float, default=0.02, help='Regularization parameter (only used for matrix factorization)')
    parser.add_argument('--log_filename', type=str, default='training_log.txt', help='Log file name')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval in epochs')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension for GNN')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for GNN training')
    
    args = parser.parse_args()
    
    main(model_type=args.model_type, K=args.K, steps=args.steps, alpha=args.alpha, beta=args.beta, log_filename=args.log_filename, eval_interval=args.eval_interval, embedding_dim=args.embedding_dim, num_epochs=args.num_epochs)

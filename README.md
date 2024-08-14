# Recommendation System on MovieLens 100K

## Project Structure

This project implements three different recommendation models: Matrix Factorization, Collaborative Filtering, and a Graph Neural Network (GNN). The code is organized into separate files for each model, with shared functionality consolidated into a common module.

### Directory Structure

```
recommendation_system/
│
├── common.py                       # Contains reused functions like data loading, logging, plotting
├── matrix_factorization.py         # Contains the matrix factorization model implementation
├── collaborative_filtering.py      # Contains the collaborative filtering model implementation
├── gnn_model.py                    # Contains the GNN model implementation
└── main.py                         # The main script to select and run the desired model
```

## How to Use the Models

### Prerequisites

Ensure you have the necessary Python packages installed. You can install the required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision torchaudio torch-geometric
```

### Running the Models

You can run the models by executing the `main.py` script and selecting the desired model type using the `--model_type` argument. Below are the instructions for running each model:

1. **Matrix Factorization**:

   ```bash
   python main.py --model_type matrix_factorization --K 30 --steps 1000 --alpha 0.001 --beta 0.01 --log_filename mf_log.txt --eval_interval 10
   ```

   - `--K`: Number of latent features (e.g., 30).
   - `--steps`: Number of iterations (e.g., 1000).
   - `--alpha`: Learning rate (e.g., 0.001).
   - `--beta`: Regularization parameter (e.g., 0.01).
   - `--log_filename`: Log file name (e.g., `mf_log.txt`).
   - `--eval_interval`: Evaluation interval (e.g., every 10 epochs).

2. **Collaborative Filtering**:

   ```bash
   python main.py --model_type collaborative_filtering --log_filename cf_log.txt --eval_interval 10
   ```

   - `--log_filename`: Log file name (e.g., `cf_log.txt`).
   - `--eval_interval`: Evaluation interval (e.g., every 10 epochs).

3. **Graph Neural Network (GNN)**:

   ```bash
   python main.py --model_type gnn --embedding_dim 16 --num_epochs 100 --log_filename gnn_log.txt --eval_interval 10
   ```

   - `--embedding_dim`: Embedding dimension (e.g., 16).
   - `--num_epochs`: Number of epochs (e.g., 100).
   - `--log_filename`: Log file name (e.g., `gnn_log.txt`).
   - `--eval_interval`: Evaluation interval (e.g., every 10 epochs).

### Log Files and Plots

- After running any model, the training and evaluation RMSE (Root Mean Squared Error) results will be logged into the specified log file.
- A plot of the RMSE over epochs will be automatically generated and saved as `training_evaluation_curve.png` in the current directory.

## How Each Model Works

### 1. **Matrix Factorization**

**Concept**: Matrix Factorization decomposes the user-item interaction matrix into two lower-dimensional matrices—one representing users and the other representing items. The dot product of these two matrices approximates the original matrix, allowing predictions of missing entries.

**Implementation**: The model iteratively updates the user and item matrices using Stochastic Gradient Descent (SGD) to minimize the difference between the predicted and actual ratings.

**Advantages**:
- Captures latent features of users and items.
- Efficient for large, sparse datasets.

### 2. **Collaborative Filtering**

**Concept**: Collaborative Filtering recommends items based on the similarity between users or items. This project uses **user-user collaborative filtering**, where users are compared to find similar users based on their past ratings. The model predicts ratings by averaging the ratings of similar users.

**Implementation**: The similarity between users is calculated using cosine similarity. Predictions are made by weighted averages of similar users' ratings.

**Advantages**:
- Simple and intuitive.
- Does not require detailed item information (content-agnostic).

### 3. **Graph Neural Network (GNN)**

**Concept**: GNNs leverage the graph structure of user-item interactions, where users and items are nodes, and interactions (ratings) are edges. The model learns embeddings for users and items by propagating and aggregating information through the graph.

**Implementation**: The model uses two layers of Graph Convolutional Networks (GCN) to learn user and item embeddings. The embeddings are used to predict ratings by taking the dot product of user and item embeddings.

**Advantages**:
- Captures complex relationships in the graph structure.
- Can incorporate additional graph-based features.

## Conclusion

This project provides a flexible framework to experiment with different recommendation models using the MovieLens 100K dataset. By separating the models into distinct modules, the code is modular and easy to extend. The main script offers a unified interface to run and evaluate any of the models, making it a valuable tool for understanding and developing recommendation systems.

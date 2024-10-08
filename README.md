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

### Bash File Explanation

1. **Log Directories**:
   - The script creates separate directories for logs (`logs/matrix_factorization`, `logs/collaborative_filtering`, and `logs/gnn`).

2. **Matrix Factorization**:
   - The script sets the parameters for matrix factorization, such as `K`, `steps`, `alpha`, and `beta`.
   - The log file is named based on these parameters (e.g., `mf_K30_steps1000_alpha0.001_beta0.01.txt`).

3. **Collaborative Filtering**:
   - The script sets parameters for collaborative filtering.
   - The log file is named based on the evaluation interval (e.g., `cf_eval_interval10.txt`).

4. **GNN**:
   - The script sets parameters for the GNN model, such as `embedding_dim` and `num_epochs`.
   - The log file is named based on these parameters (e.g., `gnn_embedding_dim16_epochs100.txt`).

5. **Running the Script**:
   - The script runs each model with the specified parameters and stores the logs in their respective folders.

### How to Use the Bash Script

1. **Save the Script**:
   - Save the script as `run_models.sh` in the root directory of your project.

2. **Make the Script Executable**:
   - Run the following command to make the script executable:
     ```bash
     chmod +x run_models.sh
     ```

3. **Execute the Script**:
   - Run the script with:
     ```bash
     ./run_models.sh
     ```

This script will run each model with the specified parameters, generate logs with descriptive filenames, and organize them into appropriate folders.


## Explanation of `matrix_factorization_step`

```python
def matrix_factorization_step(R, P, Q, K, alpha, beta):
    Q = Q.T
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                for k in range(K):
                    P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    return P, Q.T
```

#### Purpose

The `matrix_factorization_step` function performs a single iteration of the Stochastic Gradient Descent (SGD) algorithm for matrix factorization. This is used to decompose a user-item rating matrix `R` into two lower-dimensional matrices `P` (user features) and `Q` (item features), such that their product approximates `R`.

#### Parameters

- **`R`**: The original user-item rating matrix, where each entry `R[i][j]` represents the rating given by user `i` to item `j`.
- **`P`**: A matrix where each row `P[i]` represents the latent feature vector for user `i`.
- **`Q`**: A matrix where each row `Q[j]` represents the latent feature vector for item `j`. This matrix is transposed within the function for easier calculations.
- **`K`**: The number of latent features (i.e., the dimensionality of the feature vectors for users and items).
- **`alpha`**: The learning rate, which controls the size of the updates to the matrices `P` and `Q`.
- **`beta`**: The regularization parameter, which helps prevent overfitting by penalizing large values in `P` and `Q`.

#### Workflow

1. **Transpose `Q`**: 
   - The matrix `Q` is transposed to simplify the dot product calculations between user and item feature vectors.

2. **Iterate Over `R`**: 
   - The function loops over each element in the rating matrix `R`. If the rating `R[i][j]` is non-zero (indicating that user `i` has rated item `j`), it proceeds to update the corresponding rows in `P` and `Q`.

3. **Compute Error `eij`**:
   - The error `eij` is computed as the difference between the actual rating `R[i][j]` and the predicted rating (which is the dot product of `P[i,:]` and `Q[:,j]`).

4. **Update `P` and `Q`**:
   - For each latent feature `k`, the function updates `P[i][k]` and `Q[k][j]` using the gradient of the error with respect to these variables. The updates are scaled by the learning rate `alpha` and regularized by `beta` to prevent overfitting.

5. **Return Updated Matrices**:
   - The function returns the updated matrices `P` and the original shape of `Q` (by transposing it back).

## Explanation of `train_matrix_factorization`

```python
def train_matrix_factorization(train_matrix, test_matrix, K, steps, alpha, beta, log_file, eval_interval):
    num_users, num_items = train_matrix.shape
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)
    
    with open(log_file, "w") as log:
        for step in range(steps):
            P, Q = matrix_factorization_step(train_matrix, P, Q, K, alpha, beta)
            if (step + 1) % eval_interval == 0:
                train_rmse = calculate_rmse(np.dot(P, Q.T), train_matrix)
                test_rmse = calculate_rmse(np.dot(P, Q.T), test_matrix)
                log.write(f"{step + 1},{train_rmse:.4f},{test_rmse:.4f}\n")
                log.flush()
    return np.dot(P, Q.T)
```

#### Purpose

The `train_matrix_factorization` function uses the `matrix_factorization_step` function to iteratively update the matrices `P` and `Q` over multiple steps, thereby training the matrix factorization model on the provided `train_matrix`. It also evaluates the model on the `test_matrix` at specified intervals and logs the results.

#### Workflow

1. **Initialize Matrices `P` and `Q`**:
   - The matrices `P` and `Q` are initialized with random values. The dimensions are based on the number of users, items, and the specified number of latent features `K`.

2. **Open Log File**:
   - A log file is opened to record the RMSE (Root Mean Squared Error) on the training and test sets at each evaluation interval.

3. **Iterative Training**:
   - For each step in the training loop:
     - The `matrix_factorization_step` function is called to update `P` and `Q`.
     - Every `eval_interval` steps, the current model is evaluated by calculating the RMSE on both the training and test matrices.
     - The results are written to the log file.

4. **Return Final Prediction Matrix**:
   - After completing all training steps, the function returns the final prediction matrix, which is the dot product of `P` and the transpose of `Q`.

#### Key Points

- **SGD-Based Learning**: The model is trained using stochastic gradient descent, updating `P` and `Q` in a way that minimizes the reconstruction error of the original matrix `R`.
- **Regularization**: The inclusion of the regularization parameter `beta` helps prevent overfitting by penalizing overly complex models.
- **Evaluation and Logging**: The model's performance is periodically evaluated on the training and test sets, with results logged for analysis. This allows monitoring of both convergence and generalization.

### Collaborative Filtering Functions Explanation

#### 1. **`calculate_similarity(matrix)`**

```python
def calculate_similarity(matrix):
    return cosine_similarity(matrix)
```

##### Purpose
This function computes the similarity between users (or items) in the `matrix` using cosine similarity. Cosine similarity is a metric used to measure how similar two vectors are, based on the angle between them.

##### Workflow
- **Input**: 
  - `matrix`: Typically, this matrix is the user-item interaction matrix where rows represent users and columns represent items. Each element represents the rating given by a user to an item.
  
- **Process**:
  - The function computes the cosine similarity between each pair of rows (users) in the matrix. This results in a user-user similarity matrix, where each element `[i, j]` represents the cosine similarity between user `i` and user `j`.

- **Output**:
  - A square matrix of size `num_users x num_users` containing the pairwise cosine similarity between users.

##### Cosine Similarity
<img width="713" alt="Screenshot 2024-08-15 at 10 18 03 AM" src="https://github.com/user-attachments/assets/bc752a82-77b6-432a-ae79-366192e37b64">


#### 2. **`predict_ratings(user_similarity, ratings_matrix)`**

```python
def predict_ratings(user_similarity, ratings_matrix):
    user_mean = np.mean(ratings_matrix, axis=1).reshape(-1, 1)
    ratings_diff = ratings_matrix - user_mean
    pred = user_mean + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred
```

##### Purpose
This function predicts the ratings that each user would give to items they have not yet rated, based on the ratings of similar users.

##### Workflow
- **Input**:
  - `user_similarity`: The user-user similarity matrix calculated by `calculate_similarity`.
  - `ratings_matrix`: The original user-item ratings matrix.

- **Process**:
  - **User Mean Calculation**: 
    - `user_mean`: The mean rating given by each user, calculated across all the items they have rated. This is necessary to normalize the ratings.
  
  - **Difference from Mean**: 
    - `ratings_diff`: The difference between the actual ratings and the user's mean rating. This normalization helps in capturing the relative preference of a user for an item, independent of their rating scale.
  
  - **Prediction Calculation**:
    <img width="628" alt="Screenshot 2024-08-14 at 3 46 04 PM" src="https://github.com/user-attachments/assets/05bd1180-62a8-495c-9e8c-3750ac609978">


- **Output**:
  - `pred`: A matrix of predicted ratings for all users and items.

#### 3. **`train_collaborative_filtering(train_matrix, test_matrix, log_file, eval_interval)`**

```python
def train_collaborative_filtering(train_matrix, test_matrix, log_file, eval_interval):
    user_similarity = calculate_similarity(train_matrix)
    
    with open(log_file, "w") as log:
        for step in range(1, eval_interval + 1):
            predictions = predict_ratings(user_similarity, train_matrix)
            train_rmse = calculate_rmse(predictions, train_matrix)
            test_rmse = calculate_rmse(predictions, test_matrix)
            log.write(f"{step},{train_rmse:.4f},{test_rmse:.4f}\n")
            log.flush()
    return predictions
```

##### Purpose
This function trains a collaborative filtering model using user-user similarity. It evaluates the model periodically and logs the RMSE for both the training and test datasets.

##### Workflow
- **Input**:
  - `train_matrix`: The training user-item rating matrix.
  - `test_matrix`: The test user-item rating matrix.
  - `log_file`: The file where the training and test RMSE will be logged.
  - `eval_interval`: The number of steps between evaluations.

- **Process**:
  - **Similarity Calculation**: 
    - `user_similarity`: Compute the similarity between users using the `calculate_similarity` function.
  
  - **Training Loop**:
    - The model predicts ratings using `predict_ratings`.
    - The predicted ratings are evaluated against the actual ratings in both the training and test matrices using RMSE, which is logged every `eval_interval` steps.

- **Output**:
  - The function returns the final predicted ratings after training.

##### Key Points
- **User-User Collaborative Filtering**:
  - The method uses the ratings of similar users to predict ratings for each user.
  - It assumes that users with similar past ratings will continue to rate items similarly in the future.
  
- **RMSE Logging**:
  - The model's performance is logged in terms of RMSE on both training and test data, providing insight into how well the model is generalizing.

In summary, these functions together implement a user-user collaborative filtering recommendation system, where the similarity between users is used to predict how a user might rate an item they haven't rated yet. The training process involves calculating these similarities, making predictions, and then logging the performance to monitor the model's accuracy.

## GNN Model code explanation
The provided GNN (Graph Neural Network) code implements a recommendation system using a graph-based approach to model user-item interactions. Here's an explanation of the main components:

### 1. **GNNRecommendationModel Class**

```python
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
```

#### Explanation:

- **`__init__` Method**:
  - **Parameters**:
    - `num_users`: The number of unique users in the dataset.
    - `num_items`: The number of unique items (e.g., movies) in the dataset.
    - `embedding_dim`: The size of the embedding vectors for both users and items.
  - **Embedding Layers**:
    - `user_embedding`: An embedding layer that maps each user to a dense vector of size `embedding_dim`.
    - `item_embedding`: An embedding layer that maps each item to a dense vector of size `embedding_dim`.
  - **GCN Layers**:
    - `gcn1`: A Graph Convolutional Network (GCN) layer that processes the initial embeddings.
    - `gcn2`: A second GCN layer that further processes the output of `gcn1`.

- **`forward` Method**:
  - **`edge_index`**: A tensor that defines the edges in the user-item graph. Each edge connects a user node to an item node, representing a rating interaction.
  - **Embedding Concatenation**:
    - The embeddings for users and items are concatenated into a single matrix `x`, where the first `num_users` rows correspond to user embeddings, and the remaining rows correspond to item embeddings.
  - **GCN Layers**:
    - The concatenated embeddings are passed through the two GCN layers. The first GCN layer transforms the embeddings, and ReLU is applied to introduce non-linearity. The second GCN layer refines the embeddings based on the graph structure.
  - **Output**: The final embeddings for users and items, after being processed by the GCN layers.

### 2. **train_gnn Function**

```python
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
```

#### Explanation:

- **Setting Up the Graph**:
  - **`edge_index` and `edge_attr`**:
    - `edge_index`: Stores pairs of connected nodes (user-item interactions) in the graph. Each user is connected to the items they have rated, and vice versa.
    - `edge_attr`: Stores the actual rating values as edge attributes, though they are not directly used in the GCN model.

- **Model and Optimizer**:
  - **`model`**: An instance of `GNNRecommendationModel`.
  - **`optimizer`**: Uses Adam optimizer for updating the model parameters based on the gradients.
  - **`loss_fn`**: Mean Squared Error (MSE) loss function is used to measure the difference between predicted and actual ratings.

- **Training Loop**:
  - **Embeddings Calculation**: The model's `forward` method computes embeddings for users and items based on the graph structure.
  - **Prediction**: Predicted ratings are computed by taking the dot product of user and item embeddings.
  - **Loss Calculation**: The MSE loss between the predicted and actual ratings is computed only for the user-item pairs that exist in the training data (using a mask).
  - **Backward Pass and Optimization**: Gradients are calculated and used to update the model parameters.
  - **Evaluation**: Every `eval_interval` epochs, the model's performance is evaluated on both the training and test datasets using RMSE, and results are logged.

- **Return**: The trained model is returned after the training process is complete.

### Key Points

- **Graph-Based Learning**: The GNN model leverages the graph structure of user-item interactions, which is represented as edges between user and item nodes. This allows the model to capture complex relationships and dependencies in the data.
- **Two-Stage Convolution**: The GCN layers iteratively refine the embeddings by aggregating information from neighboring nodes (i.e., users aggregate information from items they have rated and vice versa).
- **Embedding Sharing**: User and item embeddings are learned jointly in a single embedding space, allowing the model to effectively capture interactions between users and items.
- **Flexibility**: The model can be trained for a specified number of epochs, and the dimensionality of embeddings can be configured.

This GNN-based recommendation approach is particularly powerful because it directly incorporates the user-item interaction graph into the learning process, allowing the model to better capture the underlying structure of the data.

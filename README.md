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

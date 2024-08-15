# collaborative_filtering.py
import numpy as np
from common import calculate_rmse
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(matrix):
    return cosine_similarity(matrix)

def predict_ratings(user_similarity, ratings_matrix):
    user_mean = np.mean(ratings_matrix, axis=1).reshape(-1, 1)
    ratings_diff = ratings_matrix - user_mean
    pred = user_mean + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred

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

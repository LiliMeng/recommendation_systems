# common.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calculate_rmse(pred, actual):
    mask = actual > 0
    return np.sqrt(np.sum((pred[mask] - actual[mask]) ** 2) / np.sum(mask))

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

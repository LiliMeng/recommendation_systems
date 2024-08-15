from common import load_movielens_data, prepare_data, plot_training_log
from matrix_factorization import train_matrix_factorization
from collaborative_filtering import train_collaborative_filtering
from gnn_model import train_gnn

def main(model_type, K, steps, alpha, beta, log_filename, plot_filename, eval_interval, embedding_dim=16, num_epochs=100):
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
    
    plot_training_log(log_filename, plot_filename)  # Pass the plot filename

if __name__ == "__main__":
    # Parsing command-line arguments and calling main
    # Ensure you parse plot_filename as well

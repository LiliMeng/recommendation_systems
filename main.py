import argparse
from common import load_movielens_data, prepare_data, plot_training_log
from matrix_factorization import train_matrix_factorization
from collaborative_filtering import train_collaborative_filtering
from gnn_model import train_gnn

def main(model_type, K, steps, alpha, beta, log_filename, plot_filename, eval_interval, embedding_dim=16, num_epochs=100):
    # Load and prepare data
    df = load_movielens_data()
    train_matrix, test_matrix, num_users, num_items = prepare_data(df)
    
    # Train model based on the selected type
    if model_type == "matrix_factorization":
        predictions = train_matrix_factorization(train_matrix, test_matrix, K, steps, alpha, beta, log_filename, eval_interval)
    elif model_type == "collaborative_filtering":
        predictions = train_collaborative_filtering(train_matrix, test_matrix, log_filename, eval_interval)
    elif model_type == "gnn":
        model = train_gnn(train_matrix, test_matrix, log_filename, eval_interval, num_epochs, embedding_dim)
    else:
        raise ValueError("Invalid model type. Choose 'matrix_factorization', 'collaborative_filtering', or 'gnn'.")
    
    # Plot the training log
    plot_training_log(log_filename, plot_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommendation System on MovieLens 100K")
    parser.add_argument('--model_type', type=str, default='matrix_factorization', choices=['matrix_factorization', 'collaborative_filtering', 'gnn'], help='Model type to use')
    parser.add_argument('--K', type=int, default=20, help='Number of latent features (only used for matrix factorization)')
    parser.add_argument('--steps', type=int, default=5000, help='Number of iterations (only used for matrix factorization)')
    parser.add_argument('--alpha', type=float, default=0.0002, help='Learning rate (only used for matrix factorization)')
    parser.add_argument('--beta', type=float, default=0.02, help='Regularization parameter (only used for matrix factorization)')
    parser.add_argument('--log_filename', type=str, required=True, help='Log file name')
    parser.add_argument('--plot_filename', type=str, required=True, help='Plot file name')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval in epochs')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension for GNN')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for GNN training')
    
    args = parser.parse_args()
    
    main(
        model_type=args.model_type,
        K=args.K,
        steps=args.steps,
        alpha=args.alpha,
        beta=args.beta,
        log_filename=args.log_filename,
        plot_filename=args.plot_filename,
        eval_interval=args.eval_interval,
        embedding_dim=args.embedding_dim,
        num_epochs=args.num_epochs
    )

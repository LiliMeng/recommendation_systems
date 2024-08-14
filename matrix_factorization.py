# matrix_factorization.py
from common import calculate_rmse

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

#!/bin/bash

# Create directories for logs if they don't exist
mkdir -p logs/matrix_factorization
mkdir -p logs/collaborative_filtering
mkdir -p logs/gnn

# Matrix Factorization Parameters
K=30
STEPS=1000
ALPHA=0.001
BETA=0.01
MF_LOG_FILENAME="logs/matrix_factorization/mf_K${K}_steps${STEPS}_alpha${ALPHA}_beta${BETA}.txt"
MF_EVAL_INTERVAL=10

# Run Matrix Factorization
echo "Running Matrix Factorization with K=${K}, steps=${STEPS}, alpha=${ALPHA}, beta=${BETA}"
python3 main.py --model_type matrix_factorization --K $K --steps $STEPS --alpha $ALPHA --beta $BETA --log_filename $MF_LOG_FILENAME --eval_interval $MF_EVAL_INTERVAL

# Collaborative Filtering Parameters
CF_LOG_FILENAME="logs/collaborative_filtering/cf_eval_interval${MF_EVAL_INTERVAL}.txt"
CF_EVAL_INTERVAL=10

# Run Collaborative Filtering
echo "Running Collaborative Filtering with eval_interval=${CF_EVAL_INTERVAL}"
python3 main.py --model_type collaborative_filtering --log_filename $CF_LOG_FILENAME --eval_interval $CF_EVAL_INTERVAL

# GNN Parameters
EMBEDDING_DIM=16
NUM_EPOCHS=100
GNN_LOG_FILENAME="logs/gnn/gnn_embedding_dim${EMBEDDING_DIM}_epochs${NUM_EPOCHS}.txt"
GNN_EVAL_INTERVAL=10

# Run GNN
echo "Running GNN with embedding_dim=${EMBEDDING_DIM}, num_epochs=${NUM_EPOCHS}"
python3 main.py --model_type gnn --embedding_dim $EMBEDDING_DIM --num_epochs $NUM_EPOCHS --log_filename $GNN_LOG_FILENAME --eval_interval $GNN_EVAL_INTERVAL

echo "All models have been run. Logs are stored in the respective folders."

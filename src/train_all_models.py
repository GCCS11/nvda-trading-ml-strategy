"""
Train all models (MLP, CNN, LSTM) and compare results.
"""

import pickle
from models.mlp import MLPModel
from models.cnn import CNNModel
from models.lstm import LSTMModel
from training import ModelTrainer
from utils import load_config
import torch


def main():
    """Train all three models and compare results."""
    import torch

    # Load config
    config = load_config()

    # Load feature columns
    with open(f"{config['data']['processed_data_path']}/feature_columns.pkl", 'rb') as f:
        feature_columns = pickle.load(f)

    input_size = len(feature_columns)
    print(f"Number of features: {input_size}")
    print("=" * 60)

    # Results storage
    results = {}

    # MLP
    print("\n" + "=" * 60)
    print("TRAINING MLP MODEL")
    print("=" * 60)
    mlp_model = MLPModel(
        input_size=input_size,
        hidden_layers=config['mlp']['hidden_layers'],
        dropout=config['mlp']['dropout']
    )
    mlp_trainer = ModelTrainer(mlp_model, config, feature_columns)
    mlp_acc, mlp_f1 = mlp_trainer.train(
        experiment_name="NVDA_Trading_Strategy",
        run_name="MLP_Model"
    )
    results['MLP'] = {'accuracy': mlp_acc, 'f1': mlp_f1, 'model': mlp_model}
    save_best_model(mlp_model, 'mlp')

    #CNN
    print("\n" + "=" * 60)
    print("TRAINING CNN MODEL")
    print("=" * 60)
    cnn_model = CNNModel(
        input_size=input_size,
        filters=config['cnn']['filters'],
        kernel_size=config['cnn']['kernel_size'],
        pool_size=config['cnn']['pool_size'],
        dropout=config['cnn']['dropout']
    )
    cnn_trainer = ModelTrainer(cnn_model, config, feature_columns)
    cnn_acc, cnn_f1 = cnn_trainer.train(
        experiment_name="NVDA_Trading_Strategy",
        run_name="CNN_Model"
    )
    results['CNN'] = {'accuracy': cnn_acc, 'f1': cnn_f1, 'model': cnn_model}
    save_best_model(cnn_model, 'cnn')

    # LSTM
    print("\n" + "=" * 60)
    print("TRAINING LSTM MODEL")
    print("=" * 60)
    lstm_model = LSTMModel(
        input_size=input_size,
        hidden_size=config['lstm']['hidden_size'],
        num_layers=config['lstm']['num_layers'],
        dropout=config['lstm']['dropout']
    )
    lstm_trainer = ModelTrainer(lstm_model, config, feature_columns)
    lstm_acc, lstm_f1 = lstm_trainer.train(
        experiment_name="NVDA_Trading_Strategy",
        run_name="LSTM_Model"
    )
    results['LSTM'] = {'accuracy': lstm_acc, 'f1': lstm_f1, 'model': lstm_model}
    save_best_model(lstm_model, 'lstm')

    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON - TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"{'Model':<10} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 60)
    for model_name in ['MLP', 'CNN', 'LSTM']:
        metrics = results[model_name]
        print(f"{model_name:<10} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f}")

    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_f1 = results[best_model_name]['f1']
    print("=" * 60)
    print(f"Best Model: {best_model_name} (F1: {best_f1:.4f})")
    print(f" Best model saved for backtesting")
    print("=" * 60)


def save_best_model(model, model_name, save_path='models/saved'):
    """Save best trained model."""
    import os
    os.makedirs(save_path, exist_ok=True)

    filepath = f"{save_path}/{model_name}_best.pth"
    torch.save(model.state_dict(), filepath)

if __name__ == "__main__":
    main()































































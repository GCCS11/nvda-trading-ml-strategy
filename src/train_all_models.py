"""
Train all models (MLP, CNN, LSTM) and compare results.
"""

import pickle
from models.mlp import MLPModel
from models.cnn import CNNModel
from models.lstm import LSTMModel
from training import ModelTrainer
from utils import load_config


def main():
    """Train all models (MLP, CNN, LSTM) and compare results."""
    config = load_config()

    # Feature columns
    with open(f"{config['data']['processed_data_path']}/feature_columns.pkl", 'rb') as f:
        feature_columns = pickle.load(f)

        input_size = len(feature_columns)
        print(f"Number of features: {input_size}")
        print("="*60)

        # Resullts storage
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
            run_name="MLPModel"
        )

        results["MLP"] = {"accuracy": mlp_acc, "f1": mlp_f1}

        # CNN
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
        results['CNN'] = {'accuracy': cnn_acc, 'f1': cnn_f1}

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
        results['LSTM'] = {'accuracy': lstm_acc, 'f1': lstm_f1}

        # Comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON - TEST SET PERFORMANCE")
        print("=" * 60)
        print(f"{'Model':<10} {'Accuracy':<12} {'F1 Score':<12}")
        print("-" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name:<10} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f}")

        # Finding the best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        print("=" * 60)
        print(f"Best Model: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")
        print("=" * 60)

if __name__ == "__main__":
    main()































































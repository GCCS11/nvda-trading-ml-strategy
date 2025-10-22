"""
Training module for deep learning models with MLFlow tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.pytorch
from utils import load_config
import os

class TradingDataset(Dataset):
    """Pytorch Dataset for trading data."""

    def __init__(self, df: pd.DataFrame, feature_columns: list):
        """
        Initialize dataset.

        Args:
            df: DataFrame with features and target.
            feature_columns: List of feature column names.
        """
        self.features = torch.FloatTensor(df[feature_columns].values)
        self.targets = torch.LongTensor(df["target"].values + 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ModelTrainer:
    """Handles model trraining with MLFlow tracking"""

    def __init__(self, model, config: dict, feature_columns: list):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train.
            config: Configuration dictionary.
            feature_columns: List of feature column names.
        """
        self.model = model
        self.config = config
        self.feature_columns = feature_columns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def load_data(self):
        """Load preprocessed data."""
        data_path = self.config["data"]["processed_data_path"]

        train_df = pd.read_csv(f"{data_path}/train.csv")
        test_df = pd.read_csv(f"{data_path}/test.csv")
        val_df = pd.read_csv(f"{data_path}/val.csv")

        return train_df, test_df, val_df

    def create_dataloaders(self, train_df, test_df, val_df):
        """Create PyTorch DataLoaders."""
        batch_size = self.config["model"]["batch_size"]

        train_dataset = TradingDataset(train_df, self.feature_columns)
        test_dataset = TradingDataset(test_df, self.feature_columns)
        val_dataset = TradingDataset(val_df, self.feature_columns)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader, val_loader

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for features, targets in train_loader:
            features, targets = features.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """Evaluate model."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets in data_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                accuracy = accuracy_score(all_targets, all_preds)
                f1 = f1_score(all_targets, all_preds, average="weighted")

                return accuracy, f1

    def train(self, experiment_name: str, run_name: str):
        """"
        Complete training loop with MlFlow tracking.

        Args:
            experiment_name: Name of the experiment.
            run_name: Name of the run.
        """
        #Load data
        train_df, test_df, val_df = self.load_data()
        train_loader, test_loader, val_loader = self.create_dataloaders(train_df, test_df, val_df)

        # MLFlw setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["model"]["learning_rate"])

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", self.model.__class__.__name__)
            mlflow.log_param("learning_rate", self.config["model"]["learning_rate"])
            mlflow.log_param("batch_size", self.config["model"]["batch_size"])
            mlflow.log_param("epochs", self.config["model"]["epochs"])

            print(f"nTRaining {run_name}")
            print("="*60)

            best_val_f1 = 0
            epochs = self.config["model"]["epochs"]

            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
                val_acc, val_f1 = self.evaluate(val_loader)

                # Metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - "
                          f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

                # Saving best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

            #Final test
            test_acc, test_f1 = self.evaluate(test_loader)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_f1", test_f1)

            print("=" * 60)
            print(f"Training complete.")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Test F1 Score: {test_f1:.4f}")
            print("=" * 60)

            # Model
            mlflow.pytorch.log_model(self.model, "model")


            return test_acc, test_f1

def main():
    import pickle
    from models.mlp import MLPModel

    #Load Config
    config = load_config()

    # Feature columns
    with open(f"{config['data']['processed_data_path']}/feature_columns.pkl", 'rb') as f:
        feature_columns = pickle.load(f)

    print(f"Number of features: {len(feature_columns)}")

    # Create MLP model
    model = MLPModel(
        input_size=len(feature_columns),
        hidden_layers=config['mlp']['hidden_layers'],
        dropout=config['mlp']['dropout']
    )

    # Train
    trainer = ModelTrainer(model, config, feature_columns)
    trainer.train(experiment_name="NVDA_Trading_Strategy", run_name="MLP_Baseline")


if __name__ == "__main__":
    main()

















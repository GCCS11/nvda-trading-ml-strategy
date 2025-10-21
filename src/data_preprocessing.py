"""
Data processing module for train / test / val splits and normalization.
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from utils import load_config


class DataPreprocessor:
    """Handles data splitting, normalization, and preparation for ML models."""

    def __init__(self, config: dict = None):
        """
        Initialize DataPreprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config else load_config()
        self.scaler = StandardScaler()
        self.feature_columns = None

    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train/test/validation sets chronologically.

        Args:
            df: DataFrame with features and target

        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        # Get split ratios
        train_ratio = self.config['data']['train_split']
        test_ratio = self.config['data']['test_split']

        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        test_end = int(n * (train_ratio + test_ratio))

        # Split chronologically (no shuffling for time series!)
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        val_df = df.iloc[test_end:].copy()

        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples ({len(train_df) / n * 100:.1f}%)")
        print(f"  Test:  {len(test_df)} samples ({len(test_df) / n * 100:.1f}%)")
        print(f"  Val:   {len(val_df)} samples ({len(val_df) / n * 100:.1f}%)")

        print(f"\nDate ranges:")
        print(f"  Train: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"  Test:  {test_df['Date'].min()} to {test_df['Date'].max()}")
        print(f"  Val:   {val_df['Date'].min()} to {val_df['Date'].max()}")

        return train_df, test_df, val_df

    def normalize_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           val_df: pd.DataFrame) -> tuple:
        """
        Normalize features using StandardScaler fitted on training data only.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            val_df: Validation DataFrame

        Returns:
            Tuple of normalized (train_df, test_df, val_df)
        """
        # Identify feature columns (exclude Date, target, forward_returns, OHLCV)
        exclude_cols = ['Date', 'target', 'forward_returns', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]

        print(f"\nNormalizing {len(self.feature_columns)} features...")

        # Fit scaler on training data only
        self.scaler.fit(train_df[self.feature_columns])

        # Transform all sets
        train_df[self.feature_columns] = self.scaler.transform(train_df[self.feature_columns])
        test_df[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])
        val_df[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns])

        print("Normalization complete")

        return train_df, test_df, val_df

    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                            val_df: pd.DataFrame):
        """
        Save processed data and scaler.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            val_df: Validation DataFrame
        """
        save_path = self.config['data']['processed_data_path']
        os.makedirs(save_path, exist_ok=True)

        # Save datasets
        train_df.to_csv(f"{save_path}/train.csv", index=False)
        test_df.to_csv(f"{save_path}/test.csv", index=False)
        val_df.to_csv(f"{save_path}/val.csv", index=False)

        # Save scaler
        with open(f"{save_path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save feature names
        with open(f"{save_path}/feature_columns.pkl", 'wb') as f:
            pickle.dump(self.feature_columns, f)

        print(f"\nProcessed data saved to {save_path}/")

    def process_pipeline(self, df: pd.DataFrame):
        """
        Complete preprocessing pipeline.

        Args:
            df: DataFrame with features and target
        """
        print("=" * 60)
        print("Starting data preprocessing pipeline...")
        print("=" * 60)

        # Split data
        train_df, test_df, val_df = self.split_data(df)

        # Normalize features
        train_df, test_df, val_df = self.normalize_features(train_df, test_df, val_df)

        # Save processed data
        self.save_processed_data(train_df, test_df, val_df)

        print("\n" + "=" * 60)
        print("Preprocessing pipeline complete!")
        print("=" * 60)


def main():
    """Test preprocessing pipeline."""
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    loader = DataLoader('NVDA')
    df = loader.load_data()

    engineer = FeatureEngineer()
    df_features = engineer.prepare_features(df)

    # Preprocess
    preprocessor = DataPreprocessor()
    preprocessor.process_pipeline(df_features)


if __name__ == "__main__":
    main()






















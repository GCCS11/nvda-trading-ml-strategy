"""
Feature engineering module for creating technical indicators.
"""

# Import Libraries
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Creates technical indicators from OHLCV data."""

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.features = []

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with return features added
        """
        # Simple returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Multi-period returns
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)

        self.features.extend(['returns', 'log_returns', 'returns_5d',
                              'returns_10d', 'returns_20d'])

        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add moving average features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with MA features added
        """
        # Simple moving averages
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()

        # Price relative to MAs
        df['price_to_sma10'] = df['Close'] / df['sma_10'] - 1
        df['price_to_sma20'] = df['Close'] / df['sma_20'] - 1
        df['price_to_sma50'] = df['Close'] / df['sma_50'] - 1

        self.features.extend(['sma_10', 'sma_20', 'sma_50',
                              'price_to_sma10', 'price_to_sma20', 'price_to_sma50'])

        return df


def main():
    """Test feature engineering on NVDA data."""
    from data_loader import DataLoader

    # Load data
    loader = DataLoader('NVDA')
    df = loader.load_data()

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    # Create features
    engineer = FeatureEngineer()
    df = engineer.add_returns(df)
    df = engineer.add_moving_averages(df)

    print(f"After feature engineering: {df.shape}")
    print(f"New features added: {len(engineer.features)}")
    print(f"Features: {engineer.features}\n")

    # Show sample
    print("Sample of new features:")
    print(df[['Date', 'Close'] + engineer.features[:5]].tail(10))

    print("\n Feature engineering test complete")


if __name__ == "__main__":
    main()




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

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility features added
        """
        # Historical volatility
        # Historical volatility (rolling std of returns)
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR (Average True Range) - simplified version
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift(1))
        df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(window=14).mean()

        # Drop intermediate calculation columns
        df.drop(['bb_middle', 'bb_std', 'high_low', 'high_close', 'low_close', 'true_range'],
                axis=1, inplace=True)

        self.features.extend(['volatility_10', 'volatility_20', 'volatility_50',
                              'bb_upper', 'bb_lower', 'bb_position', 'atr_14'])

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume based features added
        """
        # Volume moving averages
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()

        # Volume ratio
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        self.features.extend(['volume_sma_10', 'volume_sma_20', 'volume_ratio', 'obv'])

        return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with momentum features added
        """
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Rate of Change (ROC)
        df['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        self.features.extend(['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'roc_10'])

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
    df = engineer.add_volatility(df)
    df = engineer.add_volume_features(df)
    df = engineer.add_momentum_features(df)

    print(f"After feature engineering: {df.shape}")
    print(f"New features added: {len(engineer.features)}")
    print(f"Features: {engineer.features}\n")

    # Show sample
    print("Sample of new features:")
    print(df[['Date', 'Close'] + engineer.features[:5]].tail(10))

    print("\n Feature engineering test complete")


if __name__ == "__main__":
    main()




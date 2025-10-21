"""
Data loading module for NVDA trading strategy.
Downloads and validates historical price data from Yahoo Finance
"""

# Imports
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


class DataLoader:
    """Handles downloading and basic validation of stock price data."""

    def __init__(self, ticker: str, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.

        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA')
            data_dir: Directory to save raw data
        """
        self.ticker = ticker
        self.data_dir = data_dir

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def download_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Download historical dataa from Yahoo Finance.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Downloading {self.ticker} data from {start_date} to {end_date}...")

        # Download data
        df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"no data downloaded for {self.ticker}")

        # reset index to make Date a column
        df.reset_index(inplace=True)

        print(f"Downloades {len(df)} days of data")

        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the downloaded data.

        Args:
            df: Raw DataFrame from Yahoo Finance.

        Returns:
            Cleaned DataFrame
        """
        print("Validating data...")

        # Check for required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)

        if removed_rows > 0:
            print(f"Removed {removed_rows} rows with missing values")

        # Check for negative prices or zero volume
        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            print("Warning: Found non-positive prices, removing affected rows")
            df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)]

        # Sort by date
        df = df.sort_values("Date").reset_index(drop=True)

        print(f"Validation complete. Final dataset: {len(df)} rows")

        return df

    def save_data(self, df: pd.DataFrame, filename: str = None):
        """
        Save DataFrame to CSV.

        Args:
            df: DataFrame to save
            filename: Output filename (default: ticker_raw.csv)
        """
        if filename is None:
            filename = f"{self.ticker}_raw.csv"

        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load data from CSV

        Args:
            filename: CSV filename to load (default: ticker_raw.csv)

            Returns:
                DataFrame with loaded data
        """
        if filename is None:
            filename = f"{self.ticker}_raw.csv"

        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])

        print(f"Loaded {len(df)} rows from {filepath}")

        return df

    def get_data_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Print summary information about the dataset."""
        print("\n" + "=" * 50)
        print(f"Data Summary for {self.ticker}")
        print("=" * 50)
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total Trading Days: {len(df)}")
        print(f"\nPrice Statistics:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        print("=" * 50 + "\n")

def main():
    """Main function to download and save the NVDA data"""
    # Initialize loader
    loader = DataLoader("NVDA")

    # Calculate date 15 years ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Download data
    df = loader.download_data(start_date_str)

    # Validate data
    df = loader.validate_data(df)

    # Display info
    loader.get_data_info(df)

    # Save data
    loader.save_data(df)

    print("Data Loading complete.")

if __name__ == "__main__":
    main()










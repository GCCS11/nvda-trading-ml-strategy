"""
Backtesting engine for evaluating trading strategies with ML predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models.cnn import CNNModel
from models.mlp import MLPModel
import pickle
from utils import load_config
import os


class Backtester:
    """Backtests trading strategy with ML model predictions."""

    def __init__(self, model, feature_columns, config: dict = None):
        """
        Initialize backtester.

        Args:
            model: Trained PyTorch model
            feature_columns: List of feature column names
            config: Configuration dictionary
        """
        self.model = model
        self.feature_columns = feature_columns
        self.config = config if config else load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Trading parameters
        self.commission = self.config['trading']['commission']
        self.borrow_rate = self.config['trading']['borrow_rate']
        self.initial_capital = self.config['trading']['initial_capital']
        self.stop_loss = self.config['trading']['stop_loss']
        self.take_profit = self.config['trading']['take_profit']

    def predict(self, features, confidence_threshold=0.70):
        """
        Generate predictions from model with confidence filtering.

        Args:
            features: Input features
            confidence_threshold: Minimum confidence to trade (default: 0.70)
        """
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            outputs = self.model(features_tensor)

            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, predictions = probabilities.max(dim=1)

            # Convert to numpy
            max_probs = max_probs.cpu().numpy()
            predictions = predictions.cpu().numpy()

            # Apply confidence filter - set low confidence to "Hold" (0)
            low_confidence = max_probs < confidence_threshold
            predictions[low_confidence] = 1  # 1 = Hold in 0,1,2 encoding

            # Convert from 0,1,2 to -1,0,1 (short, hold, long)
            signals = predictions - 1

            return signals, max_probs

    def calculate_position_size(self, capital, price):
        """Calculate number of shares to trade."""
        # Use all available capital (simple strategy)
        shares = int(capital / price)
        return shares

    def run_backtest(self, df: pd.DataFrame, period_name: str = "Test",
                     only_long: bool = True):
        """
        Run backtest on a dataset.

        Args:
            df: DataFrame with features and price data
            period_name: Name of the period (for reporting)

        Returns:
            DataFrame with trade results and equity curve
        """
        print(f"\nRunning backtest on {period_name} period...")
        print("="*60)

        # Generate predictions
        features = df[self.feature_columns].values
        predictions, confidences = self.predict(features, confidence_threshold=0.70)

        if only_long:
            predictions[predictions == -1] = 0  # Convert shorts to holds

        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Current position: -1 (short), 0 (flat), 1 (long)
        shares = 0
        entry_price = 0
        entry_date = None

        trades = []
        equity_curve = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            current_price = row['Close']
            signal = predictions[idx]
            date = row['Date']

            # Check for stop loss or take profit if in position
            if position != 0 and shares > 0:
                pnl_pct = (current_price - entry_price) / entry_price * position

                # Stop loss hit
                if pnl_pct <= -self.stop_loss:
                    # Close position
                    exit_value = shares * current_price
                    commission_cost = exit_value * self.commission

                    if position == -1:
                        # Closing short: buy to cover
                        borrow_days = 1  # Simplified: assume 1 day
                        borrow_cost = shares * entry_price * (self.borrow_rate / 252) * borrow_days
                        pnl = shares * (entry_price - current_price) - commission_cost - borrow_cost
                    else:
                        # Closing long
                        pnl = shares * (current_price - entry_price) - commission_cost

                    capital += pnl

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'position': 'Long' if position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Stop Loss'
                    })

                    position = 0
                    shares = 0

                # Take profit hit
                elif pnl_pct >= self.take_profit:
                    # Close position
                    exit_value = shares * current_price
                    commission_cost = exit_value * self.commission

                    if position == -1:
                        borrow_days = 1
                        borrow_cost = shares * entry_price * (self.borrow_rate / 252) * borrow_days
                        pnl = shares * (entry_price - current_price) - commission_cost - borrow_cost
                    else:
                        pnl = shares * (current_price - entry_price) - commission_cost

                    capital += pnl

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'position': 'Long' if position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Take Profit'
                    })

                    position = 0
                    shares = 0

            # Process new signal
            if signal != position and capital > 0:
                # Close existing position if any
                if position != 0 and shares > 0:
                    exit_value = shares * current_price
                    commission_cost = exit_value * self.commission

                    if position == -1:
                        borrow_days = 1
                        borrow_cost = shares * entry_price * (self.borrow_rate / 252) * borrow_days
                        pnl = shares * (entry_price - current_price) - commission_cost - borrow_cost
                    else:
                        pnl = shares * (current_price - entry_price) - commission_cost

                    capital += pnl

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'position': 'Long' if position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': (current_price - entry_price) / entry_price * position,
                        'exit_reason': 'Signal Change'
                    })

                # Open new position
                if signal != 0:
                    shares = self.calculate_position_size(capital, current_price)
                    if shares > 0:
                        entry_price = current_price
                        entry_date = date
                        entry_value = shares * current_price
                        commission_cost = entry_value * self.commission
                        capital -= commission_cost
                        position = signal

            # Track equity
            current_equity = capital
            if position != 0 and shares > 0:
                # Add unrealized P&L
                if position == 1:
                    current_equity += shares * (current_price - entry_price)
                else:
                    current_equity += shares * (entry_price - current_price)

            equity_curve.append({
                'Date': date,
                'Equity': current_equity,
                'Position': position
            })

        # Close any remaining position
        if position != 0 and shares > 0:
            current_price = df.iloc[-1]['Close']
            date = df.iloc[-1]['Date']
            exit_value = shares * current_price
            commission_cost = exit_value * self.commission

            if position == -1:
                borrow_days = 1
                borrow_cost = shares * entry_price * (self.borrow_rate / 252) * borrow_days
                pnl = shares * (entry_price - current_price) - commission_cost - borrow_cost
            else:
                pnl = shares * (current_price - entry_price) - commission_cost

            capital += pnl

            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'position': 'Long' if position == 1 else 'Short',
                'entry_price': entry_price,
                'exit_price': current_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': (current_price - entry_price) / entry_price * position,
                'exit_reason': 'End of Period'
            })

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)

        return trades_df, equity_df, capital

    def calculate_metrics(self, trades_df, equity_df, final_capital):
        """Calculate performance metrics."""
        if len(trades_df) == 0:
            return {
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0
            }

        # Basic metrics
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        # Calculate returns
        equity_df['returns'] = equity_df['Equity'].pct_change()

        # Sharpe Ratio (annualized)
        if len(equity_df) > 1 and equity_df['returns'].std() > 0:
            sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Sortino Ratio (annualized)
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (equity_df['returns'].mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0

        # Max Drawdown
        equity_df['cummax'] = equity_df['Equity'].cummax()
        equity_df['drawdown'] = (equity_df['Equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()

        # Calmar Ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }

    def plot_results(self, equity_df, period_name, save_path='reports'):
        """Plot equity curve."""
        os.makedirs(save_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(equity_df['Date'], equity_df['Equity'], linewidth=2, label='Strategy Equity')
        ax.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'Equity Curve - {period_name} Period', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/equity_curve_{period_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Equity curve saved to {save_path}/equity_curve_{period_name.lower()}.png")

    def print_results(self, metrics, period_name, final_capital):
        """Print backtest results."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - {period_name.upper()} PERIOD")
        print(f"{'='*60}")
        print(f"Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"Final Capital:      ${final_capital:,.2f}")
        print(f"Total Return:       {metrics['total_return']*100:.2f}%")
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:     {metrics['total_trades']}")
        print(f"  Winning Trades:   {metrics['winning_trades']}")
        print(f"  Losing Trades:    {metrics['losing_trades']}")
        print(f"  Win Rate:         {metrics['win_rate']*100:.2f}%")
        print(f"  Avg Win:          ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss:         ${metrics['avg_loss']:,.2f}")
        print(f"\nPerformance Metrics:")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:    {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
        print(f"  Calmar Ratio:     {metrics['calmar_ratio']:.2f}")
        print(f"{'='*60}\n")


def main():
    """Run backtest with best model (MLP based on latest results)."""
    print("="*60)
    print("BACKTESTING WITH TRAINED MODEL")
    print("="*60)

    # Load config
    config = load_config()

    # Load feature columns
    data_path = config['data']['processed_data_path']
    with open(f"{data_path}/feature_columns.pkl", 'rb') as f:
        feature_columns = pickle.load(f)

    # Load MLP model (best performer)
    model = MLPModel(
        input_size=len(feature_columns),
        hidden_layers=config['mlp']['hidden_layers'],
        dropout=config['mlp']['dropout']
    )

    # Load trained weights
    model.load_state_dict(torch.load('models/saved/mlp_best.pth'))
    print("Loaded trained MLP model")

    # Initialize backtester
    backtester = Backtester(model, feature_columns, config)

    # Load datasets with price data
    test_df = pd.read_csv(f"{data_path}/test.csv")
    val_df = pd.read_csv(f"{data_path}/val.csv")

    test_df['Date'] = pd.to_datetime(test_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])

    # Run backtests
    for df, period_name in [(test_df, 'Test'), (val_df, 'Validation')]:
        trades_df, equity_df, final_capital = backtester.run_backtest(
            df, period_name, only_long=True  # <-- Add this
        )
        metrics = backtester.calculate_metrics(trades_df, equity_df, final_capital)
        backtester.print_results(metrics, period_name, final_capital)
        backtester.plot_results(equity_df, period_name)

    print("Backtesting complete.")


if __name__ == "__main__":
    main()

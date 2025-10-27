"""
Compare different trading strategies to find the best approach.
"""

import pandas as pd
import torch
from models.mlp import MLPModel
import pickle
from utils import load_config
from backtesting import Backtester


def compare_strategies():
    """Compare long-only vs smart shorts vs all signals."""
    print("=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    # Load config and setup
    config = load_config()
    data_path = config['data']['processed_data_path']

    with open(f"{data_path}/feature_columns.pkl", 'rb') as f:
        feature_columns = pickle.load(f)

    # Load model
    model = MLPModel(
        input_size=len(feature_columns),
        hidden_layers=config['mlp']['hidden_layers'],
        dropout=config['mlp']['dropout']
    )
    model.load_state_dict(torch.load('models/saved/mlp_best.pth'))
    print("✓ Loaded trained MLP model\n")

    # Load data
    test_df = pd.read_csv(f"{data_path}/test.csv")
    val_df = pd.read_csv(f"{data_path}/val.csv")
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])

    # Test strategies
    strategies = ['long_only', 'smart', 'all_signals']
    strategy_names = {
        'long_only': 'Long Only (No Shorts)',
        'smart': 'Smart (Long 70%, Short 80%)',
        'all_signals': 'All Signals (70% threshold)'
    }

    results = {}

    for strategy in strategies:
        print("\n" + "=" * 60)
        print(f"TESTING: {strategy_names[strategy]}")
        print("=" * 60)

        backtester = Backtester(model, feature_columns, config)

        strategy_results = {}

        for period_name, df in [('test', test_df), ('val', val_df)]:
            trades_df, equity_df, final_capital = backtester.run_backtest(
                df, period_name, strategy_mode=strategy
            )
            metrics = backtester.calculate_metrics(trades_df, equity_df, final_capital)

            strategy_results[period_name] = {
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe_ratio'],
                'trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'max_dd': metrics['max_drawdown'],
                'final_capital': final_capital
            }

        results[strategy] = strategy_results

    # Print comparison table
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 60)

    for period in ['test', 'val']:
        print(f"\n{period.upper()} PERIOD:")
        print(f"{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Trades':<10} {'Win Rate':<10}")
        print("-" * 70)

        for strategy in strategies:
            r = results[strategy][period]
            print(f"{strategy_names[strategy]:<30} "
                  f"{r['return'] * 100:>10.2f}%  "
                  f"{r['sharpe']:>8.2f}  "
                  f"{r['trades']:>8d}  "
                  f"{r['win_rate'] * 100:>8.1f}%")

    # Recommend best strategy
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    # Score strategies (weighted: return 40%, sharpe 30%, win_rate 30%)
    scores = {}
    for strategy in strategies:
        test_r = results[strategy]['test']
        val_r = results[strategy]['val']

        # Average metrics across test and val
        avg_return = (test_r['return'] + val_r['return']) / 2
        avg_sharpe = (test_r['sharpe'] + val_r['sharpe']) / 2
        avg_win_rate = (test_r['win_rate'] + val_r['win_rate']) / 2

        # Normalize and weight
        score = (avg_return * 0.4) + (avg_sharpe * 0.3) + (avg_win_rate * 0.3)
        scores[strategy] = score

    best_strategy = max(scores, key=scores.get)

    print(f"\nBest Strategy: {strategy_names[best_strategy]}")
    print(f"\nTest Return:  {results[best_strategy]['test']['return'] * 100:.2f}%")
    print(f"Val Return:   {results[best_strategy]['val']['return'] * 100:.2f}%")
    print(
        f"Avg Sharpe:   {(results[best_strategy]['test']['sharpe'] + results[best_strategy]['val']['sharpe']) / 2:.2f}")

    print("\n" + "=" * 60)

    return best_strategy, results


if __name__ == "__main__":
    best, results = compare_strategies()
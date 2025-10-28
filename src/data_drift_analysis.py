"""
Data drift analysis to detect distribution changes across train/test/val periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import load_config
import os


class DataDriftAnalyzer:
    """Analyzes data drift across different time periods."""

    def __init__(self, config: dict = None):
        """Initialize analyzer."""
        self.config = config if config else load_config()
        self.drift_results = {}

    def load_datasets(self):
        """Load train/test/val datasets."""
        data_path = self.config['data']['processed_data_path']

        train_df = pd.read_csv(f"{data_path}/train.csv")
        test_df = pd.read_csv(f"{data_path}/test.csv")
        val_df = pd.read_csv(f"{data_path}/val.csv")

        # Convert date columns
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        val_df['Date'] = pd.to_datetime(val_df['Date'])

        return train_df, test_df, val_df

    def ks_test(self, feature: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                val_df: pd.DataFrame) -> dict:
        """
        Perform Kolmogorov-Smirnov test for distribution drift.

        Args:
            feature: Feature name
            train_df, test_df, val_df: DataFrames

        Returns:
            Dictionary with test results
        """
        # KS test: train vs test
        ks_stat_test, p_value_test = stats.ks_2samp(
            train_df[feature].dropna(),
            test_df[feature].dropna()
        )

        # KS test: train vs val
        ks_stat_val, p_value_val = stats.ks_2samp(
            train_df[feature].dropna(),
            val_df[feature].dropna()
        )

        # Drift detected if p-value < 0.05
        drift_test = p_value_test < 0.05
        drift_val = p_value_val < 0.05

        return {
            'feature': feature,
            'ks_stat_test': ks_stat_test,
            'p_value_test': p_value_test,
            'drift_test': drift_test,
            'ks_stat_val': ks_stat_val,
            'p_value_val': p_value_val,
            'drift_val': drift_val
        }

    def analyze_all_features(self, train_df, test_df, val_df, feature_columns):
        """Analyze drift for all features."""
        results = []

        print("Analyzing data drift for all features...")
        print("=" * 60)

        for feature in feature_columns:
            result = self.ks_test(feature, train_df, test_df, val_df)
            results.append(result)

        drift_df = pd.DataFrame(results)
        self.drift_results = drift_df

        # Summary statistics
        n_drift_test = drift_df['drift_test'].sum()
        n_drift_val = drift_df['drift_val'].sum()

        print(f"\nDrift Detection Summary:")
        print(f"  Features with drift (Train vs Test): {n_drift_test}/{len(feature_columns)}")
        print(f"  Features with drift (Train vs Val):  {n_drift_val}/{len(feature_columns)}")
        print("=" * 60)

        return drift_df

    def plot_drift_summary(self, drift_df, save_path='reports'):
        """Create drift summary visualization."""
        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: P-values for Train vs Test
        drift_test = drift_df.sort_values('p_value_test')
        colors = ['red' if x else 'green' for x in drift_test['drift_test']]

        axes[0].barh(range(len(drift_test)), drift_test['p_value_test'], color=colors, alpha=0.6)
        axes[0].axvline(x=0.05, color='black', linestyle='--', label='p=0.05 threshold')
        axes[0].set_yticks(range(len(drift_test)))
        axes[0].set_yticklabels(drift_test['feature'], fontsize=8)
        axes[0].set_xlabel('P-value')
        axes[0].set_title('Data Drift: Train vs Test')
        axes[0].legend()
        axes[0].invert_yaxis()

        # Plot 2: P-values for Train vs Val
        drift_val = drift_df.sort_values('p_value_val')
        colors = ['red' if x else 'green' for x in drift_val['drift_val']]

        axes[1].barh(range(len(drift_val)), drift_val['p_value_val'], color=colors, alpha=0.6)
        axes[1].axvline(x=0.05, color='black', linestyle='--', label='p=0.05 threshold')
        axes[1].set_yticks(range(len(drift_val)))
        axes[1].set_yticklabels(drift_val['feature'], fontsize=8)
        axes[1].set_xlabel('P-value')
        axes[1].set_title('Data Drift: Train vs Val')
        axes[1].legend()
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"{save_path}/drift_summary.png", dpi=300, bbox_inches='tight')
        print(f"\n Drift summary saved to {save_path}/drift_summary.png")
        plt.close()

    def plot_feature_distributions(self, feature: str, train_df, test_df, val_df,
                                   save_path='reports'):
        """Plot distribution of a single feature across periods."""
        os.makedirs(save_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot distributions
        ax.hist(train_df[feature].dropna(), bins=50, alpha=0.5, label='Train', density=True)
        ax.hist(test_df[feature].dropna(), bins=50, alpha=0.5, label='Test', density=True)
        ax.hist(val_df[feature].dropna(), bins=50, alpha=0.5, label='Val', density=True)

        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature} Across Periods')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/dist_{feature}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_top_drifted_features(self, train_df, test_df, val_df, n=5, save_path='reports'):
        """Plot distributions for top N drifted features."""
        # Get top N drifted features based on val p-value
        top_drifted = self.drift_results.nsmallest(n, 'p_value_val')

        print(f"\nTop {n} drifted features (Train vs Val):")
        print(top_drifted[['feature', 'p_value_val', 'drift_val']])

        for _, row in top_drifted.iterrows():
            feature = row['feature']
            self.plot_feature_distributions(feature, train_df, test_df, val_df, save_path)

        print(f"\n Distribution plots saved to {save_path}/")

    def generate_report(self, save_path='reports'):
        """Generate drift analysis report."""
        os.makedirs(save_path, exist_ok=True)

        # Save drift statistics to CSV
        self.drift_results.to_csv(f"{save_path}/drift_statistics.csv", index=False)
        print(f"\n Drift statistics saved to {save_path}/drift_statistics.csv")

        # Create summary text file
        with open(f"{save_path}/drift_summary.txt", 'w') as f:
            f.write("DATA DRIFT ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write("Features with Significant Drift (p < 0.05):\n")
            f.write("-" * 60 + "\n\n")

            f.write("Train vs Test:\n")
            drifted_test = self.drift_results[self.drift_results['drift_test']]
            f.write(f"  {len(drifted_test)} features\n")
            for _, row in drifted_test.iterrows():
                f.write(f"    - {row['feature']}: p-value = {row['p_value_test']:.4f}\n")

            f.write("\nTrain vs Val:\n")
            drifted_val = self.drift_results[self.drift_results['drift_val']]
            f.write(f"  {len(drifted_val)} features\n")
            for _, row in drifted_val.iterrows():
                f.write(f"    - {row['feature']}: p-value = {row['p_value_val']:.4f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("\nInterpretation:\n")
            f.write("Features with p-value < 0.05 show significant distribution changes.\n")
            f.write("This may indicate:\n")
            f.write("  - Market regime changes\n")
            f.write("  - Volatility shifts\n")
            f.write("  - Structural breaks in the data\n")
            f.write("  - Need for model retraining or adaptation\n")

        print(f" Drift summary saved to {save_path}/drift_summary.txt")


def main():
    """Run complete drift analysis."""
    import pickle

    print("=" * 60)
    print("DATA DRIFT ANALYSIS")
    print("=" * 60)

    # Initialize
    analyzer = DataDriftAnalyzer()
    config = load_config()

    # Load data
    print("\nLoading datasets...")
    train_df, test_df, val_df = analyzer.load_datasets()

    # Load feature columns
    with open(f"{config['data']['processed_data_path']}/feature_columns.pkl", 'rb') as f:
        feature_columns = pickle.load(f)

    print(f"Loaded {len(feature_columns)} features")

    # Analyze drift
    drift_df = analyzer.analyze_all_features(train_df, test_df, val_df, feature_columns)

    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_drift_summary(drift_df)
    analyzer.plot_top_drifted_features(train_df, test_df, val_df, n=5)

    # Generate report
    analyzer.generate_report()

    print("\n" + "=" * 60)
    print("Data drift analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
Analyze model predictions to understand behavior.
"""

import pandas as pd
import numpy as np
import torch
from models.mlp import MLPModel
import pickle
from utils import load_config
import matplotlib.pyplot as plt


def analyze_predictions():
    """Analyze what the model is actually predicting."""
    # Load config and data
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
    model.eval()

    # Load test data
    test_df = pd.read_csv(f"{data_path}/test.csv")

    # Get predictions with probabilities
    features = torch.FloatTensor(test_df[feature_columns].values)

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()
        predictions = outputs.argmax(dim=1).numpy()

    # Convert predictions: 0,1,2 -> -1,0,1
    signals = predictions - 1

    # Analyze
    print("=" * 60)
    print("MODEL PREDICTION ANALYSIS")
    print("=" * 60)

    print("\nSignal Distribution:")
    unique, counts = np.unique(signals, return_counts=True)
    for signal, count in zip(unique, counts):
        signal_name = {-1: 'Short', 0: 'Hold', 1: 'Long'}[signal]
        pct = count / len(signals) * 100
        print(f"  {signal_name:6s}: {count:4d} ({pct:5.1f}%)")

    print("\nPrediction Confidence (Max Probability):")
    max_probs = probabilities.max(axis=1)
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  Median confidence: {np.median(max_probs):.3f}")
    print(f"  Min confidence: {max_probs.min():.3f}")
    print(f"  Max confidence: {max_probs.max():.3f}")

    print("\nConfidence Distribution:")
    print(f"  < 40%: {(max_probs < 0.4).sum()} predictions")
    print(f"  40-60%: {((max_probs >= 0.4) & (max_probs < 0.6)).sum()} predictions")
    print(f"  60-80%: {((max_probs >= 0.6) & (max_probs < 0.8)).sum()} predictions")
    print(f"  > 80%: {(max_probs >= 0.8).sum()} predictions")

    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Random (50%)')
    plt.axvline(x=0.6, color='g', linestyle='--', label='Proposed threshold (60%)')
    plt.xlabel('Prediction Confidence (Max Probability)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Prediction Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/prediction_confidence.png', dpi=300, bbox_inches='tight')
    print("\nConfidence distribution saved to reports/prediction_confidence.png")

    print("=" * 60)

    return signals, max_probs


if __name__ == "__main__":
    analyze_predictions()
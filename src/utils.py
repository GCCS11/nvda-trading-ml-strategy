"""
Utility functions for the tradinig strategy project.
"""

# Import libraries
import yaml
import os


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration parameters
    """
    if config_path is None:
        # Get the project root directory (parent of src)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Test config loading."""
    config = load_config()

    print("Configuration loaded successfully!")
    print("\nData config:")
    print(f"  Ticker: {config['data']['ticker']}")
    print(
        f"  Train/Test/Val: {config['data']['train_split']}/{config['data']['test_split']}/{config['data']['val_split']}")

    print("\nModel config:")
    print(f"  Batch size: {config['model']['batch_size']}")
    print(f"  Epochs: {config['model']['epochs']}")
    print(f"  Learning rate: {config['model']['learning_rate']}")

    print("\nMLP architecture:")
    print(f"  Hidden layers: {config['mlp']['hidden_layers']}")
    print(f"  Dropout: {config['mlp']['dropout']}")

    print("\nConfig test complete!")

if __name__ == "__main__":
    main()

















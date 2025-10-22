"""
Long Short-Term Memory (LSTM) model for trading signal prediction.
"""

# Import libraries
import torch
import torch.nn as nn
from numpy.distutils.lib2def import output_def


class LSTMModel(nn.Module):
    """
    LSTM model for capturing long-term temporal dependencies.
    Processes sequences of features over time.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, num_classes: int = 3):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape for LSTM: (batch, seq_len, features)
        # For single timestep, seq_len = 1
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take output from last timestep
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layers
        output = self.fc(lstm_out)

        return output

def main():
    """Testing LSTM model."""
    input_size = 27
    hidden_size = 64
    num_layers = 2
    batch_size = 32

    # Model
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,)

    print("LSTM Model Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)

    #Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Dummy
    dummy_input = torch.randn(batch_size, input_size)
    output = model(dummy_input)

    print(f"\nForward Pass Test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    print("\nLSTM model test complete.")


if __name__ == "__main__":
    main()


































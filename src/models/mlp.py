"""
Multii-Layer Perceptron (MLP) model for trading signal prediction.
"""

#Import libraries
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    Simple MLP baseline model.
    Learns feature combinations without temporal structure.
    """

    def __init__(self, input_size: int, hidden_layers: list, dropout: float = 0.3,
                 num_classes: int = 3):
        """
        Initiiliaze MLP model.

        Args:
            input size: Number of input features.
            hidden layers: List of hidden layers.
            dropout: Dropout rate for regularization.
            num_classes: Number of output classes.(3: lonog/hold/short)
        """
        super(MLPModel, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def main(self):
        """Test MLP model creation."""
        # Test parameters
        input_size = 27
        hidden_layers = [128, 64, 32]
        batch_size = 32

        # Create model
        model = MLPModel(input_size=input_size, hidden_layers=hidden_layers)

        print("MLP Model Architecture:")
        print("=" * 60)
        print(model)
        print("=" * 60)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test forward pass
        dummy_input = torch.randn(batch_size, input_size)
        output = model(dummy_input)

        print(f"\nForward Pass Test:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0]}")

        print("\n MLP model test complete.")

if __name__ == "__main__":
    main()





































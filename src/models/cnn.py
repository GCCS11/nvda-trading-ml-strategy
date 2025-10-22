"""
Convolutional Neural Network (CNN) model for trading signal prediction
"""

# Libraries
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    CNN model for capturing local temporal patterns.
    """

    def __init__(self, input_size: int, filters: list, kernel_size: int = 3,
                 pool_size: int = 2, dropout: float = 0.3, num_classes: int = 3):
        """
        Initialize CNN model.

        Args:
            input_size: Number of input features
            filters: List of filter sizes [64, 128, 64]
            kernel_size: Convolution kernel size
            pool_size: Max pooling size
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super(CNNModel, self).__init__()

        self.input_size = input_size

        #Convulutional layers
        conv_layers = []
        in_channels = 1

        for out_channels in filters:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # calculate flattened size for convolutions.
        self.flatten_size = filters[-1] * (input_size // (pool_size ** len(filters)))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def main():
    """Test CNN model creation."""
    # Test parameters
    input_size = 27
    filters = [64, 128, 64]
    batch_size = 32

    # Create model
    model = CNNModel(input_size=input_size, filters=filters)

    print("CNN Model Architecture:")
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

    print("\nCNN model test complete.")


if __name__ == "__main__":
    main()











































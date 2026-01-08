"""
Model definition for Dataset Cartography.

This module creates an EfficientNetB0 model for binary classification
with appropriate modifications for our task.
"""

import torch
import torch.nn as nn
from torchvision import models

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import DROPOUT_RATE, NUM_CLASSES


def create_model(pretrained=True, dropout_rate=None):
    """
    Create an EfficientNetB0 model for binary classification.
    
    Modifications from base model:
    - Replace final classifier with custom head
    - Add dropout layer for regularization
    - Single output neuron for binary classification (with BCEWithLogitsLoss)
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout probability (default from config)
        
    Returns:
        Modified EfficientNetB0 model
    """
    if dropout_rate is None:
        dropout_rate = DROPOUT_RATE
    
    # Load pretrained EfficientNetB0
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        print("Loaded EfficientNetB0 with ImageNet pretrained weights")
    else:
        model = models.efficientnet_b0(weights=None)
        print("Created EfficientNetB0 without pretrained weights")
    
    # Get the number of input features to the classifier
    num_features = model.classifier[1].in_features
    
    # Replace classifier with custom head:
    # Dropout -> Linear (binary classification)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )
    
    print(f"Modified classifier: Dropout({dropout_rate}) -> Linear({num_features}, {NUM_CLASSES})")
    
    return model


def get_device():
    """
    Get the best available device (GPU if available, else CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def count_parameters(model):
    """
    Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return trainable, total


def save_model(model, filepath):
    """
    Save model weights to file.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath, device=None):
    """
    Load model weights from file.
    
    Args:
        filepath: Path to the saved model
        device: Device to load the model to
        
    Returns:
        Model with loaded weights
    """
    if device is None:
        device = get_device()
    
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {filepath}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model module...")
    print("=" * 50)
    
    # Create model
    model = create_model()
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    
    # Test forward pass
    device = get_device()
    model = model.to(device)
    
    # Create dummy input (batch of 4 images, 3 channels, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.squeeze().tolist()}")
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output)
    print(f"  Probabilities: {probs.squeeze().tolist()}")
    
    print("\nModel test complete!")

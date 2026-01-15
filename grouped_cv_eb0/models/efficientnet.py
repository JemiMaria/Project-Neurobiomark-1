"""
EfficientNetB0 model for ALS classification.

Architecture:
- EfficientNetB0 backbone (ImageNet pretrained)
- Custom classifier head: Dropout + Linear(1280 → 1)
- Single logit output for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PRETRAINED, DROPOUT_RATE, NUM_CLASSES


def create_efficientnet_b0(pretrained=None, dropout_rate=None, num_classes=None):
    """
    Create EfficientNetB0 model with custom classifier head.
    
    Args:
        pretrained: Use ImageNet pretrained weights (default from config)
        dropout_rate: Dropout rate for classifier (default from config)
        num_classes: Number of output classes (default 1 for binary)
        
    Returns:
        nn.Module: EfficientNetB0 model
    """
    if pretrained is None:
        pretrained = PRETRAINED
    if dropout_rate is None:
        dropout_rate = DROPOUT_RATE
    if num_classes is None:
        num_classes = NUM_CLASSES
    
    # Load pretrained EfficientNetB0
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        print("  ✓ Loaded EfficientNetB0 with ImageNet pretrained weights")
    else:
        model = models.efficientnet_b0(weights=None)
        print("  ✓ Loaded EfficientNetB0 (random initialization)")
    
    # Get number of features from original classifier
    in_features = model.classifier[1].in_features  # 1280 for EfficientNetB0
    
    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    print(f"  ✓ Custom classifier: Dropout({dropout_rate}) → Linear({in_features} → {num_classes})")
    
    return model


def get_model_info(model):
    """
    Get model architecture information.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Model information including parameter counts
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count backbone vs classifier parameters
    backbone_params = 0
    classifier_params = 0
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params += param.numel()
        else:
            backbone_params += param.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'backbone_params': backbone_params,
        'classifier_params': classifier_params,
        'total_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }


def freeze_backbone(model):
    """
    Freeze backbone parameters (features), keep classifier trainable.
    
    Used in Phase A of training (head-only training).
    
    Args:
        model: EfficientNetB0 model
    """
    for param in model.features.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_last_stage(model):
    """
    Unfreeze last stage of backbone for fine-tuning.
    
    Used in Phase B of training.
    EfficientNetB0 has 9 blocks (0-8), we unfreeze block 8.
    
    Args:
        model: EfficientNetB0 model
    """
    # Unfreeze last block (block 8)
    for param in model.features[8].parameters():
        param.requires_grad = True
    
    # Also unfreeze BatchNorm in last block
    for module in model.features[8].modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Last stage unfrozen. Trainable params: {trainable:,}")


def unfreeze_all(model):
    """
    Unfreeze all parameters.
    
    Args:
        model: EfficientNetB0 model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ All parameters unfrozen. Trainable params: {trainable:,}")


def get_parameter_groups(model, head_lr, backbone_lr_multiplier):
    """
    Get parameter groups with different learning rates.
    
    - Classifier head: head_lr
    - Backbone: head_lr * backbone_lr_multiplier
    
    Args:
        model: EfficientNetB0 model
        head_lr: Learning rate for classifier head
        backbone_lr_multiplier: Multiplier for backbone LR
        
    Returns:
        list: Parameter groups for optimizer
    """
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    param_groups = [
        {'params': classifier_params, 'lr': head_lr, 'name': 'classifier'},
        {'params': backbone_params, 'lr': head_lr * backbone_lr_multiplier, 'name': 'backbone'},
    ]
    
    print(f"  ✓ Parameter groups: classifier LR={head_lr:.2e}, backbone LR={head_lr * backbone_lr_multiplier:.2e}")
    
    return param_groups

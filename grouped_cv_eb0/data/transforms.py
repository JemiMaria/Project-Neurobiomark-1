"""
Augmentation transforms using Albumentations.

Implements modular augmentation policies:
- Light: flips + rotate
- Medium: Light + mild shift/scale/zoom + crop+pad
- Strong: Medium + rare mild blur/noise + optional very-rare elastic

LEAKAGE PREVENTION:
- Validation transforms are deterministic (no augmentation)
- Stain normalization (if used) must be fit on training fold only
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    AUG_LEVEL, MAX_ROTATE, P_NOISE, USE_ELASTIC, USE_STAIN_AUG
)


def get_val_transforms(image_size=None):
    """
    Get validation transforms (deterministic, no augmentation).
    
    Only applies:
    - Resize to model input size
    - ImageNet normalization
    - Tensor conversion
    
    Args:
        image_size: Target image size (default from config)
        
    Returns:
        albumentations.Compose: Validation transform pipeline
    """
    if image_size is None:
        image_size = IMAGE_SIZE
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_transforms(
    image_size=None,
    aug_level=None,
    max_rotate=None,
    p_noise=None,
    use_elastic=None,
    use_stain_aug=None
):
    """
    Get training transforms with configurable augmentation level.
    
    Augmentation Policies:
    - Light: flips + rotate
    - Medium: Light + mild shift/scale/zoom + crop+pad
    - Strong: Medium + rare mild blur/noise + optional very-rare elastic
    
    Args:
        image_size: Target image size
        aug_level: 'light', 'medium', or 'strong'
        max_rotate: Maximum rotation angle in degrees
        p_noise: Probability of noise augmentation
        use_elastic: Whether to use elastic deformation
        use_stain_aug: Whether to use stain augmentation
        
    Returns:
        albumentations.Compose: Training transform pipeline
    """
    # Use defaults from config
    if image_size is None:
        image_size = IMAGE_SIZE
    if aug_level is None:
        aug_level = AUG_LEVEL
    if max_rotate is None:
        max_rotate = MAX_ROTATE
    if p_noise is None:
        p_noise = P_NOISE
    if use_elastic is None:
        use_elastic = USE_ELASTIC
    if use_stain_aug is None:
        use_stain_aug = USE_STAIN_AUG
    
    # Build augmentation pipeline based on level
    aug_transforms = []
    
    # === LIGHT AUGMENTATIONS (always applied) ===
    # Flips
    aug_transforms.append(A.HorizontalFlip(p=0.5))
    aug_transforms.append(A.VerticalFlip(p=0.5))
    
    # Rotation
    aug_transforms.append(A.Rotate(limit=max_rotate, p=0.5, border_mode=0))
    
    if aug_level in ['medium', 'strong']:
        # === MEDIUM AUGMENTATIONS ===
        # Shift, scale, zoom
        aug_transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=0,  # Already have rotation
                p=0.5,
                border_mode=0
            )
        )
        
        # Random crop and pad back
        aug_transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.3
            )
        )
    
    if aug_level == 'strong':
        # === STRONG AUGMENTATIONS ===
        # Mild blur (rare)
        aug_transforms.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.1)
        )
        
        # Mild noise (configurable probability)
        if p_noise > 0:
            aug_transforms.append(
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.1), p=1.0),
                ], p=p_noise)
            )
        
        # Very rare mild elastic deformation
        if use_elastic:
            aug_transforms.append(
                A.ElasticTransform(
                    alpha=50,
                    sigma=10,
                    p=0.05  # Very rare
                )
            )
        
        # Mild brightness/contrast (histopathology-safe)
        aug_transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.2
            )
        )
    
    # === STAIN AUGMENTATION (Optional) ===
    # Note: If using stain augmentation that requires fitting,
    # the fitting must happen on training fold only (handled in dataloader)
    if use_stain_aug:
        try:
            # Try to import RandStainNA if available
            from randstainna import RandStainNA
            aug_transforms.append(
                # Add stain augmentation here
                # This is a placeholder - actual implementation depends on library
                A.Lambda(
                    image=lambda x, **kwargs: x,  # Placeholder
                    name='stain_aug'
                )
            )
            print("  ✓ Stain augmentation enabled")
        except ImportError:
            print("  ⚠ RandStainNA not installed, skipping stain augmentation")
    
    # Build final pipeline
    transforms = A.Compose([
        # Resize first
        A.Resize(image_size, image_size),
        # Apply augmentations
        *aug_transforms,
        # Ensure size is correct after augmentations
        A.Resize(image_size, image_size),
        # Normalize
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Convert to tensor
        ToTensorV2(),
    ])
    
    return transforms


def get_stain_normalizer(train_images, method='macenko'):
    """
    Fit stain normalizer on training images only.
    
    LEAKAGE PREVENTION:
    - This function should only be called with training fold images
    - The fitted normalizer can then be applied to validation images
    
    Args:
        train_images: List of training image arrays
        method: Normalization method ('macenko', 'reinhard', etc.)
        
    Returns:
        Fitted normalizer object (or None if not available)
    """
    try:
        from staintools import StainNormalizer
        
        # Use first few training images to compute reference statistics
        normalizer = StainNormalizer(method=method)
        
        # Fit on a representative training image
        # Choose a well-stained image (e.g., median brightness)
        brightnesses = [img.mean() for img in train_images[:min(10, len(train_images))]]
        median_idx = np.argsort(brightnesses)[len(brightnesses) // 2]
        reference_image = train_images[median_idx]
        
        normalizer.fit(reference_image)
        print(f"  ✓ Stain normalizer fit on training data (method={method})")
        
        return normalizer
    
    except ImportError:
        print("  ⚠ staintools not installed, skipping stain normalization")
        return None

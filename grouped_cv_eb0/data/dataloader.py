"""
Dataloader creation with patient-level balanced sampling.

LEAKAGE PREVENTION:
- Dataloaders are created AFTER fold splits are determined
- Train/val datasets use fold-specific patient lists
- Balanced sampling is at patient level, not image level

IMPORTANT:
- pos_weight is NOT used (sampler replaces it)
- Validation uses natural sampling (no balancing)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

from .dataset import ALSDataset
from .transforms import get_train_transforms, get_val_transforms


def create_patient_balanced_sampler(dataset):
    """
    Create a patient-level balanced sampler for training.
    
    BALANCING LOGIC:
    - Each patient should have equal probability of being sampled
    - Within each class (ALS/Control), patients are balanced
    - ALS and Control classes are also balanced
    
    This prevents:
    - Oversampling patients with many images
    - Class imbalance in training batches
    
    WHY PATIENT-LEVEL (not image-level):
    - Images from same patient are correlated
    - Image-level balancing would oversample patients with fewer images
    - Patient-level ensures each patient contributes equally
    
    Args:
        dataset: ALSDataset with patient information
        
    Returns:
        WeightedRandomSampler: Sampler for DataLoader
    """
    patient_info = dataset.get_patient_info()
    patient_to_indices = patient_info['patient_to_indices']
    patient_to_label = patient_info['patient_to_label']
    
    # Count patients per class
    n_patients_class0 = sum(1 for p, l in patient_to_label.items() if l == 0)
    n_patients_class1 = sum(1 for p, l in patient_to_label.items() if l == 1)
    n_total_patients = n_patients_class0 + n_patients_class1
    
    # Calculate weight for each sample
    # Weight = (1 / n_images_in_patient) × (1 / n_patients_in_class)
    # This ensures:
    # - Each patient has equal total weight (regardless of image count)
    # - Each class has equal total weight
    
    sample_weights = np.zeros(len(dataset))
    
    for patient_id, indices in patient_to_indices.items():
        label = patient_to_label[patient_id]
        n_images = len(indices)
        n_patients_in_class = n_patients_class1 if label == 1 else n_patients_class0
        
        # Weight per image for this patient
        weight = (1.0 / n_images) * (1.0 / n_patients_in_class)
        
        for idx in indices:
            sample_weights[idx] = weight
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(dataset),
        replacement=True  # Allow resampling to balance classes
    )
    
    # Print summary
    print(f"\n  Patient-balanced sampler created:")
    print(f"    - Patients: {n_total_patients} (ALS: {n_patients_class1}, Control: {n_patients_class0})")
    print(f"    - Samples: {len(dataset)}")
    print(f"    - Each patient has equal expected contribution per epoch")
    
    return sampler


def create_fold_dataloaders(
    metadata_df,
    fold_splits,
    fold_id,
    batch_size=None,
    num_workers=None,
    train_transform=None,
    val_transform=None,
    use_balanced_sampler=True
):
    """
    Create train and validation dataloaders for a specific fold.
    
    LEAKAGE PREVENTION:
    - Uses pre-computed fold splits (no patient overlap)
    - Train and val datasets are created from fold-specific patient lists
    - Balanced sampling only on training data
    
    Args:
        metadata_df: Full metadata DataFrame
        fold_splits: Dict from create_patient_grouped_splits
        fold_id: Which fold to create dataloaders for
        batch_size: Batch size (default from config)
        num_workers: Number of workers (default from config)
        train_transform: Training transforms (default: get_train_transforms())
        val_transform: Validation transforms (default: get_val_transforms())
        use_balanced_sampler: Whether to use patient-balanced sampling
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()
    
    # Get patient lists for this fold
    train_patients = fold_splits[fold_id]['train_patients']
    val_patients = fold_splits[fold_id]['val_patients']
    
    # Verify no overlap (defensive check)
    overlap = set(train_patients) & set(val_patients)
    if overlap:
        raise RuntimeError(f"LEAKAGE DETECTED: Fold {fold_id} has patient overlap!")
    
    print(f"\nCreating dataloaders for Fold {fold_id}:")
    print(f"  Train patients: {len(train_patients)}")
    print(f"  Val patients: {len(val_patients)}")
    
    # Create datasets with fold-specific patient lists
    train_dataset = ALSDataset(
        metadata_df=metadata_df,
        patient_ids=train_patients,
        transform=train_transform
    )
    
    val_dataset = ALSDataset(
        metadata_df=metadata_df,
        patient_ids=val_patients,
        transform=val_transform
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create train loader with balanced sampler
    if use_balanced_sampler:
        train_sampler = create_patient_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # Uses sampler, not shuffle
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=True
        )
    
    # Create val loader with natural sampling (no balancing)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic for reproducibility
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

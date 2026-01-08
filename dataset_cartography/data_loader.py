"""
Data loading utilities for Dataset Cartography.

This module handles:
- Reading metadata from Excel file
- Loading and preprocessing images
- Creating PyTorch datasets and dataloaders
- Handling class imbalance with weighted sampling
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    IMAGE_FOLDER, EXCEL_FILE, IMAGE_SIZE, 
    IMAGENET_MEAN, IMAGENET_STD,
    COL_IMAGE_NO, COL_CASE_ID, COL_CONDITION,
    POSITIVE_CLASS, NEGATIVE_CLASS,
    VALIDATION_SPLIT, TEST_SPLIT, BATCH_SIZE
)


def extract_patient_id(case_id):
    """
    Extract Patient ID from Case ID.
    
    Example: "SD028-12 Concord BA46" -> "SD028-12"
    
    Args:
        case_id: Full case identifier string
        
    Returns:
        Patient ID (part before first space)
    """
    if pd.isna(case_id):
        return "Unknown"
    return str(case_id).split()[0]


def load_metadata(excel_path=None):
    """
    Load and process metadata from Excel file.
    
    Args:
        excel_path: Path to Excel file (uses config default if None)
        
    Returns:
        DataFrame with columns: image_no, case_id, patient_id, condition, label
    """
    if excel_path is None:
        excel_path = EXCEL_FILE
    
    print(f"Loading metadata from: {excel_path}")
    
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Create processed dataframe
    metadata = pd.DataFrame({
        'image_no': df[COL_IMAGE_NO].astype(int),
        'case_id': df[COL_CASE_ID].astype(str),
        'condition': df[COL_CONDITION].astype(str)
    })
    
    # Extract patient ID from case ID
    metadata['patient_id'] = metadata['case_id'].apply(extract_patient_id)
    
    # Convert condition to binary label (Case=1, Control=0)
    metadata['label'] = (metadata['condition'] == POSITIVE_CLASS).astype(int)
    
    # Print summary
    print(f"  Total samples: {len(metadata)}")
    print(f"  {POSITIVE_CLASS} (label=1): {(metadata['label'] == 1).sum()}")
    print(f"  {NEGATIVE_CLASS} (label=0): {(metadata['label'] == 0).sum()}")
    print(f"  Unique patients: {metadata['patient_id'].nunique()}")
    
    return metadata


def get_image_transforms():
    """
    Get image preprocessing transforms.
    
    No augmentation for cartography analysis - just resize and normalize.
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # ImageNet normalization
    ])
    return transform


class BrainTissueDataset(Dataset):
    """
    PyTorch Dataset for brain tissue images.
    
    Loads images based on image number and applies preprocessing transforms.
    """
    
    def __init__(self, metadata, image_folder=None, transform=None):
        """
        Initialize dataset.
        
        Args:
            metadata: DataFrame with image_no and label columns
            image_folder: Path to folder containing .tiff images
            transform: Image transforms to apply
        """
        self.metadata = metadata.reset_index(drop=True)
        self.image_folder = image_folder if image_folder else IMAGE_FOLDER
        self.transform = transform if transform else get_image_transforms()
        
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            tuple: (image_tensor, label, image_no)
        """
        row = self.metadata.iloc[idx]
        image_no = int(row['image_no'])
        label = int(row['label'])
        
        # Construct image path: e.g., "1.tiff"
        image_path = os.path.join(self.image_folder, f"{image_no}.tiff")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, image_no


def create_weighted_sampler(labels):
    """
    Create a weighted random sampler to balance classes in mini-batches.
    
    Args:
        labels: Array or list of class labels
        
    Returns:
        WeightedRandomSampler object
    """
    labels = np.array(labels)
    
    # Count samples per class
    class_counts = np.bincount(labels)
    
    # Calculate weight for each class (inverse of frequency)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample based on its class
    sample_weights = class_weights[labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    
    print(f"  Class counts: {dict(enumerate(class_counts))}")
    print(f"  Class weights: {dict(enumerate(class_weights))}")
    
    return sampler


def split_data(metadata, validation_split=None, random_seed=42):
    """
    Split metadata into training and validation sets at PATIENT level.
    
    This prevents data leakage by ensuring all samples from the same patient
    stay together in one split. No patient appears in multiple splits.
    
    Args:
        metadata: DataFrame with labels and patient_id
        validation_split: Fraction of PATIENTS for validation (default from config)
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_metadata, val_metadata)
    """
    if validation_split is None:
        validation_split = VALIDATION_SPLIT
    
    # Get unique patients
    patients = metadata['patient_id'].unique()
    n_patients = len(patients)
    
    print(f"\nPatient-level data split (seed={random_seed}):")
    print(f"  Total patients: {n_patients}")
    
    # Split patients (not samples) into train and validation
    train_patients, val_patients = train_test_split(
        patients,
        test_size=validation_split,
        random_state=random_seed
    )
    
    # Get all samples belonging to train/val patients
    train_meta = metadata[metadata['patient_id'].isin(train_patients)].copy()
    val_meta = metadata[metadata['patient_id'].isin(val_patients)].copy()
    
    # Print split summary
    print(f"  Train patients: {len(train_patients)} -> {len(train_meta)} samples")
    print(f"  Val patients: {len(val_patients)} -> {len(val_meta)} samples")
    
    # Verify no patient leakage
    train_patient_set = set(train_meta['patient_id'].unique())
    val_patient_set = set(val_meta['patient_id'].unique())
    overlap = train_patient_set.intersection(val_patient_set)
    
    if overlap:
        raise ValueError(f"Data leakage detected! Patients in both splits: {overlap}")
    else:
        print("  ✓ No patient leakage (verified)")
    
    # Print class distribution per split
    print(f"  Train class distribution: Case={train_meta['label'].sum()}, Control={(train_meta['label']==0).sum()}")
    print(f"  Val class distribution: Case={val_meta['label'].sum()}, Control={(val_meta['label']==0).sum()}")
    
    return train_meta, val_meta


def split_data_with_test(metadata, validation_split=None, test_split=None, random_seed=42):
    """
    Split metadata into train/validation/test sets at PATIENT level.
    
    This prevents data leakage by ensuring all samples from the same patient
    stay together in one split. No patient appears in multiple splits.
    
    Args:
        metadata: DataFrame with labels and patient_id
        validation_split: Fraction of PATIENTS for validation (default from config)
        test_split: Fraction of PATIENTS for test (default from config)
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_metadata, val_metadata, test_metadata)
    """
    if validation_split is None:
        validation_split = VALIDATION_SPLIT
    if test_split is None:
        test_split = TEST_SPLIT
    
    # Get unique patients
    patients = metadata['patient_id'].unique()
    n_patients = len(patients)
    
    print(f"\nPatient-level data split (seed={random_seed}):")
    print(f"  Total patients: {n_patients}")
    
    # First split: separate test patients
    remaining_patients, test_patients = train_test_split(
        patients,
        test_size=test_split,
        random_state=random_seed
    )
    
    # Second split: separate train and val from remaining
    # Adjust validation split to account for removed test patients
    adjusted_val_split = validation_split / (1 - test_split)
    train_patients, val_patients = train_test_split(
        remaining_patients,
        test_size=adjusted_val_split,
        random_state=random_seed
    )
    
    # Get all samples belonging to each patient group
    train_meta = metadata[metadata['patient_id'].isin(train_patients)].copy()
    val_meta = metadata[metadata['patient_id'].isin(val_patients)].copy()
    test_meta = metadata[metadata['patient_id'].isin(test_patients)].copy()
    
    # Print split summary
    print(f"  Train patients: {len(train_patients)} -> {len(train_meta)} samples")
    print(f"  Val patients: {len(val_patients)} -> {len(val_meta)} samples")
    print(f"  Test patients: {len(test_patients)} -> {len(test_meta)} samples")
    
    # Verify no patient leakage
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    
    if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
        raise ValueError("Data leakage detected! Patients appear in multiple splits.")
    else:
        print("  ✓ No patient leakage (verified)")
    
    # Print class distribution per split
    print(f"  Train class distribution: Case={train_meta['label'].sum()}, Control={(train_meta['label']==0).sum()}")
    print(f"  Val class distribution: Case={val_meta['label'].sum()}, Control={(val_meta['label']==0).sum()}")
    print(f"  Test class distribution: Case={test_meta['label'].sum()}, Control={(test_meta['label']==0).sum()}")
    
    return train_meta, val_meta, test_meta


def create_dataloaders(train_metadata, val_metadata, batch_size=None, use_weighted_sampler=True):
    """
    Create training and validation dataloaders.
    
    Args:
        train_metadata: Training set metadata
        val_metadata: Validation set metadata
        batch_size: Batch size (default from config)
        use_weighted_sampler: Whether to use weighted sampling for training
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # Create datasets
    train_dataset = BrainTissueDataset(train_metadata)
    val_dataset = BrainTissueDataset(val_metadata)
    
    # Create training dataloader with weighted sampler
    if use_weighted_sampler:
        print("Creating weighted sampler for training:")
        sampler = create_weighted_sampler(train_metadata['label'].values)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    # Create validation dataloader (no sampling needed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_full_dataset_loader(metadata, batch_size=None):
    """
    Create a dataloader for the full dataset (for computing metrics).
    
    No shuffling or weighted sampling - just iterate through all samples.
    
    Args:
        metadata: Full dataset metadata
        batch_size: Batch size (default from config)
        
    Returns:
        DataLoader object
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    dataset = BrainTissueDataset(metadata)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order consistent
        num_workers=0,
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading module...")
    print("=" * 50)
    
    # Load metadata
    try:
        metadata = load_metadata()
        print("\nFirst few rows of metadata:")
        print(metadata.head(10))
        
        # Test split
        train_meta, val_meta = split_data(metadata, random_seed=42)
        
        # Test dataloader creation
        print("\nCreating dataloaders...")
        train_loader, val_loader = create_dataloaders(train_meta, val_meta)
        
        # Test loading a batch
        print("\nLoading first batch...")
        images, labels, image_nos = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Image numbers: {image_nos.tolist()}")
        
        print("\nData loading test complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update IMAGE_FOLDER and EXCEL_FILE in config.py")

"""
Patient-Grouped Cross-Validation Splits

Creates stratified group k-fold splits where:
- Groups = patient_id (no patient in both train and val)
- Stratification by patient-level labels

LEAKAGE PREVENTION:
- Splits are created BEFORE dataset/dataloader creation
- Ensures complete patient separation between folds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_FOLDS, RANDOM_SEED, SPLITS_DIR


def create_patient_grouped_splits(metadata_df, n_folds=None, random_seed=None, save=True):
    """
    Create patient-grouped stratified k-fold splits.
    
    LEAKAGE PREVENTION:
    - Uses StratifiedGroupKFold to ensure no patient overlap
    - Stratifies by patient-level label (not image-level)
    - Saves splits before any training begins
    
    Args:
        metadata_df: DataFrame with [image_path, patient_id, y_true]
        n_folds: Number of folds (default from config)
        random_seed: Random seed (default from config)
        save: Whether to save splits to CSV
        
    Returns:
        dict: {fold_id: {'train_patients': [...], 'val_patients': [...]}}
    """
    if n_folds is None:
        n_folds = N_FOLDS
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    print(f"\n{'='*60}")
    print("CREATING PATIENT-GROUPED CV SPLITS")
    print(f"{'='*60}")
    
    # Get unique patients and their labels
    patient_df = metadata_df.groupby('patient_id').agg({
        'y_true': 'first'  # One label per patient (validated earlier)
    }).reset_index()
    
    patients = patient_df['patient_id'].values
    labels = patient_df['y_true'].values
    
    print(f"\nTotal patients: {len(patients)}")
    print(f"  ALS (y=1): {labels.sum()}")
    print(f"  Control (y=0): {len(labels) - labels.sum()}")
    print(f"  Folds: {n_folds}")
    
    # Create stratified group k-fold splitter
    # Note: groups=patients ensures no patient overlap
    # Stratification ensures class balance across folds
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    splits = {}
    split_records = []
    
    print(f"\nFold distribution:")
    print("-" * 50)
    
    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(patients, labels, groups=patients)):
        train_patients = patients[train_idx].tolist()
        val_patients = patients[val_idx].tolist()
        
        # Verify no overlap (sanity check)
        overlap = set(train_patients) & set(val_patients)
        if overlap:
            raise RuntimeError(f"LEAKAGE DETECTED: Fold {fold_id} has patient overlap: {overlap}")
        
        splits[fold_id] = {
            'train_patients': train_patients,
            'val_patients': val_patients
        }
        
        # Get class distribution for this fold
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        print(f"  Fold {fold_id}: Train {len(train_patients)} patients "
              f"(ALS:{train_labels.sum()}, Ctrl:{len(train_labels)-train_labels.sum()}) | "
              f"Val {len(val_patients)} patients "
              f"(ALS:{val_labels.sum()}, Ctrl:{len(val_labels)-val_labels.sum()})")
        
        # Record for saving
        for pid in train_patients:
            split_records.append({
                'patient_id': pid,
                'fold_id': fold_id,
                'split': 'train',
                'y_true': patient_df[patient_df['patient_id'] == pid]['y_true'].values[0]
            })
        for pid in val_patients:
            split_records.append({
                'patient_id': pid,
                'fold_id': fold_id,
                'split': 'val',
                'y_true': patient_df[patient_df['patient_id'] == pid]['y_true'].values[0]
            })
    
    # Save splits
    if save:
        splits_df = pd.DataFrame(split_records)
        splits_path = SPLITS_DIR / "grouped_cv_folds.csv"
        splits_df.to_csv(splits_path, index=False)
        print(f"\n  ✓ Saved splits to: {splits_path}")
    
    print(f"\n{'='*60}")
    
    return splits


def load_splits(splits_path=None):
    """
    Load pre-computed splits from CSV.
    
    Args:
        splits_path: Path to splits CSV (default: grouped_cv_folds.csv)
        
    Returns:
        dict: {fold_id: {'train_patients': [...], 'val_patients': [...]}}
    """
    if splits_path is None:
        splits_path = SPLITS_DIR / "grouped_cv_folds.csv"
    
    splits_df = pd.read_csv(splits_path)
    
    splits = {}
    for fold_id in splits_df['fold_id'].unique():
        fold_data = splits_df[splits_df['fold_id'] == fold_id]
        splits[fold_id] = {
            'train_patients': fold_data[fold_data['split'] == 'train']['patient_id'].tolist(),
            'val_patients': fold_data[fold_data['split'] == 'val']['patient_id'].tolist()
        }
    
    return splits


def verify_no_leakage(splits):
    """
    Verify that no patient appears in both train and val for any fold.
    
    Args:
        splits: dict from create_patient_grouped_splits or load_splits
        
    Returns:
        bool: True if no leakage detected
        
    Raises:
        RuntimeError: If leakage detected
    """
    print("\nVerifying no patient leakage...")
    
    for fold_id, fold_splits in splits.items():
        train_set = set(fold_splits['train_patients'])
        val_set = set(fold_splits['val_patients'])
        
        overlap = train_set & val_set
        if overlap:
            raise RuntimeError(f"LEAKAGE: Fold {fold_id} has overlap: {overlap}")
    
    print("  ✓ No patient leakage detected across all folds")
    return True

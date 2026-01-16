"""
LOPO (Leave-One-Patient-Out) splits for ALS vs ALS-FTD.

Creates patient-level cross-validation splits where each patient
is held out as test set exactly once.

For ~10 patients, this creates ~10 folds.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import LOPO_DIR


def create_lopo_splits(metadata_df, output_dir=None):
    """
    Create Leave-One-Patient-Out (LOPO) splits.
    
    For each unique patient_id:
    - hold_out_patient = test fold
    - remaining = train fold
    
    Args:
        metadata_df: DataFrame with columns [image_path, patient_id, y_true, category]
        output_dir: Directory to save splits CSV
        
    Returns:
        dict: {fold_id: {'train_patients': [...], 'test_patient': patient_id, 'y_true': label}}
    """
    if output_dir is None:
        output_dir = LOPO_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CREATING LOPO SPLITS")
    print(f"{'='*60}")
    
    # Get unique patients with their labels
    patient_labels = metadata_df.groupby('patient_id')['y_true'].first()
    all_patients = patient_labels.index.tolist()
    n_patients = len(all_patients)
    
    print(f"\n  Total patients: {n_patients}")
    print(f"  LOPO folds: {n_patients} (one patient held out per fold)")
    
    # Create LOPO splits
    splits = {}
    split_records = []
    
    for fold_id, test_patient in enumerate(all_patients):
        # Train patients = all except test
        train_patients = [p for p in all_patients if p != test_patient]
        test_label = patient_labels[test_patient]
        
        splits[fold_id] = {
            'train_patients': train_patients,
            'test_patient': test_patient,
            'y_true': int(test_label)
        }
        
        # Record for CSV
        split_records.append({
            'fold_id': fold_id,
            'patient_id': test_patient,
            'role': 'test',
            'y_true': int(test_label)
        })
        
        for tp in train_patients:
            split_records.append({
                'fold_id': fold_id,
                'patient_id': tp,
                'role': 'train',
                'y_true': int(patient_labels[tp])
            })
    
    # Validate no overlap
    print("\n  Validating splits...")
    for fold_id, split_data in splits.items():
        overlap = set(split_data['train_patients']) & {split_data['test_patient']}
        if overlap:
            raise RuntimeError(f"LEAKAGE DETECTED: Fold {fold_id} has patient overlap!")
    print("  ✓ No patient overlap between train/test per fold")
    
    # Save splits
    splits_df = pd.DataFrame(split_records)
    splits_path = output_dir / "lopo_splits.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"\n  ✓ Splits saved to: {splits_path}")
    
    # Print fold distribution
    print(f"\n  Fold distribution:")
    for fold_id, split_data in splits.items():
        test_patient = split_data['test_patient']
        test_label = "ALS-FTD" if split_data['y_true'] == 1 else "ALS"
        n_train = len(split_data['train_patients'])
        print(f"    Fold {fold_id}: Test={test_patient} ({test_label}), Train={n_train} patients")
    
    return splits


def load_lopo_splits(splits_path=None):
    """
    Load LOPO splits from CSV.
    
    Args:
        splits_path: Path to lopo_splits.csv
        
    Returns:
        dict: {fold_id: {'train_patients': [...], 'test_patient': patient_id, 'y_true': label}}
    """
    if splits_path is None:
        splits_path = LOPO_DIR / "lopo_splits.csv"
    
    splits_df = pd.read_csv(splits_path)
    
    splits = {}
    for fold_id in splits_df['fold_id'].unique():
        fold_data = splits_df[splits_df['fold_id'] == fold_id]
        
        test_row = fold_data[fold_data['role'] == 'test'].iloc[0]
        train_patients = fold_data[fold_data['role'] == 'train']['patient_id'].tolist()
        
        splits[fold_id] = {
            'train_patients': train_patients,
            'test_patient': test_row['patient_id'],
            'y_true': int(test_row['y_true'])
        }
    
    return splits


def get_fold_data(metadata_df, splits, fold_id):
    """
    Get train and test data for a specific fold.
    
    Args:
        metadata_df: Full metadata DataFrame
        splits: LOPO splits dictionary
        fold_id: Which fold to get
        
    Returns:
        tuple: (train_df, test_df, test_patient_id)
    """
    split_data = splits[fold_id]
    
    train_patients = split_data['train_patients']
    test_patient = split_data['test_patient']
    
    train_df = metadata_df[metadata_df['patient_id'].isin(train_patients)].copy()
    test_df = metadata_df[metadata_df['patient_id'] == test_patient].copy()
    
    return train_df, test_df, test_patient

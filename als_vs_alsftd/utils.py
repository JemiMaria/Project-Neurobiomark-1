"""
Utility functions for ALS vs ALS-FTD pipeline.

Includes:
- Data loading and filtering
- Patient ID extraction
- Seed setting
- Wilson confidence interval computation
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats

from config import (
    METADATA_PATH, IMAGE_DIR,
    COL_IMAGE_NO, COL_CASE_ID, COL_CATEGORY, COL_CONDITION,
    POSITIVE_CATEGORY, NEGATIVE_CATEGORY, POSITIVE_LABEL, NEGATIVE_LABEL,
    INCLUDED_CATEGORIES, RANDOM_SEEDS
)


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def load_and_filter_als_vs_alsftd(metadata_path=None):
    """
    Load and filter metadata for ALS vs ALS-FTD classification.
    
    Filters:
    - Keep only rows where category ∈ {Concordant, Discordant}
    - Remove all controls
    - Map labels: Concordant → y=1 (ALS-FTD), Discordant → y=0 (ALS)
    
    Validates:
    - No missing patient_id
    - No missing images
    - No inconsistent labels per patient
    
    Args:
        metadata_path: Path to metadata Excel file
        
    Returns:
        pd.DataFrame with columns [image_path, patient_id, y_true, category]
    """
    if metadata_path is None:
        metadata_path = METADATA_PATH
    
    print(f"\n{'='*60}")
    print("LOADING AND FILTERING DATA: ALS vs ALS-FTD")
    print(f"{'='*60}")
    print(f"\nLoading metadata from: {metadata_path}")
    
    # Load Excel file
    df = pd.read_excel(metadata_path, engine='openpyxl')
    print(f"  Raw rows loaded: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Check required columns exist
    required_cols = [COL_IMAGE_NO, COL_CASE_ID, COL_CATEGORY]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Show category distribution before filtering
    print(f"\n  Category distribution (before filtering):")
    for cat, count in df[COL_CATEGORY].value_counts().items():
        print(f"    {cat}: {count}")
    
    # Filter to include only ALS and ALS-FTD (exclude controls)
    df_filtered = df[df[COL_CATEGORY].isin(INCLUDED_CATEGORIES)].copy()
    print(f"\n  After filtering (excluding controls): {len(df_filtered)} rows")
    
    if len(df_filtered) == 0:
        raise ValueError(f"No data remaining after filtering. "
                        f"Check that categories {INCLUDED_CATEGORIES} exist in data.")
    
    # Create processed DataFrame
    processed = pd.DataFrame({
        'image_no': df_filtered[COL_IMAGE_NO].astype(int),
        'case_id': df_filtered[COL_CASE_ID].astype(str),
        'category': df_filtered[COL_CATEGORY].astype(str)
    })
    
    # Extract patient_id
    processed['patient_id'] = processed['case_id'].apply(extract_patient_id)
    
    # Map labels
    # Concordant → ALS-FTD → y=1 (positive)
    # Discordant → ALS → y=0 (negative)
    processed['y_true'] = processed['category'].map({
        POSITIVE_CATEGORY: POSITIVE_LABEL,
        NEGATIVE_CATEGORY: NEGATIVE_LABEL
    })
    
    # Create image_path from image_no (e.g., 1 -> "1.tif")
    processed['image_path'] = processed['image_no'].apply(lambda x: f"{x}.tif")
    
    # Validation 1: No missing patient_id
    missing_patient = processed['patient_id'].isna().sum()
    unknown_patient = (processed['patient_id'] == "Unknown").sum()
    if missing_patient > 0 or unknown_patient > 0:
        raise ValueError(f"Found {missing_patient + unknown_patient} samples with missing patient_id")
    print(f"\n  ✓ No missing patient_id")
    
    # Validation 2: Check for missing images
    missing_images = []
    for idx, row in processed.iterrows():
        img_path = IMAGE_DIR / row['image_path']
        if not img_path.exists():
            missing_images.append(str(row['image_path']))
    
    if missing_images:
        print(f"  ⚠ Warning: {len(missing_images)} images not found in {IMAGE_DIR}")
        if len(missing_images) <= 5:
            for p in missing_images:
                print(f"    - {p}")
    else:
        print(f"  ✓ All images exist")
    
    # Validation 3: One consistent label per patient_id
    patient_labels = processed.groupby('patient_id')['y_true'].nunique()
    inconsistent = patient_labels[patient_labels > 1]
    if len(inconsistent) > 0:
        raise ValueError(f"Inconsistent labels for patients: {inconsistent.index.tolist()}")
    print(f"  ✓ Consistent labels per patient")
    
    # Summary statistics
    n_patients = processed['patient_id'].nunique()
    patient_label_df = processed.groupby('patient_id')['y_true'].first()
    n_positive = (patient_label_df == POSITIVE_LABEL).sum()
    n_negative = (patient_label_df == NEGATIVE_LABEL).sum()
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY: ALS vs ALS-FTD")
    print(f"{'='*60}")
    print(f"  Total patients: {n_patients}")
    print(f"    - ALS-FTD (Concordant, y=1): {n_positive} patients")
    print(f"    - ALS (Discordant, y=0): {n_negative} patients")
    print(f"  Total images: {len(processed)}")
    print(f"    - ALS-FTD images: {(processed['y_true'] == POSITIVE_LABEL).sum()}")
    print(f"    - ALS images: {(processed['y_true'] == NEGATIVE_LABEL).sum()}")
    print(f"  Images per patient: {len(processed) / n_patients:.1f} (avg)")
    
    # Check balance
    balance_ratio = n_positive / n_negative if n_negative > 0 else float('inf')
    if 0.8 <= balance_ratio <= 1.2:
        print(f"  ✓ Classes are balanced (ratio: {balance_ratio:.2f})")
    else:
        print(f"  ⚠ Classes are imbalanced (ratio: {balance_ratio:.2f})")
    
    return processed[['image_path', 'patient_id', 'y_true', 'category']]


def compute_wilson_confidence_interval(successes, total, confidence=0.95):
    """
    Compute Wilson score confidence interval for a proportion.
    
    Better than normal approximation for small sample sizes (N~10).
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        tuple: (lower_bound, upper_bound, point_estimate)
    """
    if total == 0:
        return (0.0, 1.0, 0.5)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return (lower, upper, p)


def get_device_info():
    """Print device information."""
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  Device: CPU")

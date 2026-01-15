"""
Dataset class for ALS classification.

Handles image loading with proper validation to prevent data leakage.
Dataset is created from fold-specific patient lists only.

Uses same image_keys.xlsx format as LOPO pipeline:
- Image No: image number (1, 2, 3, ...)
- Case ID: e.g., "SD028-12 Concord BA46"
- Condition: "Case" (ALS=1) or "Control" (0)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    IMAGE_DIR, METADATA_PATH,
    COL_IMAGE_NO, COL_CASE_ID, COL_CONDITION,
    POSITIVE_CLASS, NEGATIVE_CLASS
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


def load_and_validate_metadata(metadata_path=None):
    """
    Load and validate metadata from Excel file (image_keys.xlsx).
    
    Uses same format as LOPO pipeline:
    - Image No: image number (1, 2, 3, ...)
    - Case ID: e.g., "SD028-12 Concord BA46"
    - Condition: "Case" (ALS=1) or "Control" (0)
    
    Validates:
    - No missing patient_id
    - No missing images
    - One consistent label per patient_id
    
    Args:
        metadata_path: Path to Excel file. If None, uses config default.
        
    Returns:
        pd.DataFrame: Validated metadata with columns [image_path, patient_id, y_true]
        
    Raises:
        ValueError: If validation fails
    """
    if metadata_path is None:
        metadata_path = METADATA_PATH
    
    print(f"\nLoading metadata from: {metadata_path}")
    
    # Load Excel file
    df = pd.read_excel(metadata_path, engine='openpyxl')
    print(f"  Raw columns: {df.columns.tolist()}")
    
    # Validate required columns exist
    required_excel_cols = [COL_IMAGE_NO, COL_CASE_ID, COL_CONDITION]
    missing_cols = [c for c in required_excel_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Available columns: {df.columns.tolist()}")
    
    # Create processed DataFrame matching expected format
    processed = pd.DataFrame({
        'image_no': df[COL_IMAGE_NO].astype(int),
        'case_id': df[COL_CASE_ID].astype(str),
        'condition': df[COL_CONDITION].astype(str)
    })
    
    # Extract patient_id from Case ID (e.g., "SD028-12 Concord BA46" -> "SD028-12")
    processed['patient_id'] = processed['case_id'].apply(extract_patient_id)
    
    # Convert condition to binary label (Case=1, Control=0)
    processed['y_true'] = (processed['condition'] == POSITIVE_CLASS).astype(int)
    
    # Create image_path from image_no (e.g., 1 -> "1.tif")
    processed['image_path'] = processed['image_no'].apply(lambda x: f"{x}.tif")
    
    print(f"  Total samples: {len(processed)}")
    print(f"  {POSITIVE_CLASS} (ALS=1): {(processed['y_true'] == 1).sum()}")
    print(f"  {NEGATIVE_CLASS} (Control=0): {(processed['y_true'] == 0).sum()}")
    
    # Validation 1: No missing patient_id
    missing_patient = processed['patient_id'].isna().sum()
    unknown_patient = (processed['patient_id'] == "Unknown").sum()
    if missing_patient > 0 or unknown_patient > 0:
        print(f"  ⚠ Warning: {missing_patient + unknown_patient} samples with missing/unknown patient_id")
    else:
        print(f"  ✓ No missing patient_id")
    
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
            for p in missing_images[:3]:
                print(f"    - {p}")
            print(f"    ... and {len(missing_images) - 3} more")
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
    n_cases = processed.groupby('patient_id')['y_true'].first().sum()
    n_controls = n_patients - n_cases
    
    print(f"\nDataset Summary:")
    print(f"  Patients: {n_patients} (ALS: {int(n_cases)}, Control: {int(n_controls)})")
    print(f"  Images: {len(processed)}")
    print(f"  Images per patient: {len(processed) / n_patients:.1f} (avg)")
    
    # Return with standardized column names
    return processed[['image_path', 'patient_id', 'y_true']]


class ALSDataset(Dataset):
    """
    PyTorch Dataset for ALS classification.
    
    LEAKAGE PREVENTION:
    - Dataset is created from a specific patient list (not all data)
    - This ensures train/val datasets contain only fold-specific patients
    
    Args:
        metadata_df: DataFrame with [image_path, patient_id, y_true]
        patient_ids: List of patient IDs to include (for fold-specific datasets)
        transform: Albumentations transform pipeline
        image_dir: Base directory for images (if paths are relative)
    """
    
    def __init__(self, metadata_df, patient_ids, transform=None, image_dir=None):
        """
        Initialize dataset with fold-specific patient list.
        
        CRITICAL: Only samples from specified patient_ids are included.
        This prevents patient overlap between train/val.
        """
        self.image_dir = Path(image_dir) if image_dir else IMAGE_DIR
        self.transform = transform
        
        # Filter to only specified patients (LEAKAGE PREVENTION)
        self.df = metadata_df[metadata_df['patient_id'].isin(patient_ids)].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples found for patient_ids: {patient_ids[:5]}...")
        
        # Store patient-level info for balanced sampling
        self.patient_ids = list(patient_ids)
        self.patient_to_indices = self._build_patient_index()
        self.patient_to_label = self._get_patient_labels()
    
    def _build_patient_index(self):
        """Build mapping from patient_id to sample indices."""
        patient_to_indices = {}
        for idx, row in self.df.iterrows():
            pid = row['patient_id']
            if pid not in patient_to_indices:
                patient_to_indices[pid] = []
            patient_to_indices[pid].append(idx)
        return patient_to_indices
    
    def _get_patient_labels(self):
        """Get label for each patient."""
        return self.df.groupby('patient_id')['y_true'].first().to_dict()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = Path(row['image_path'])
        if not img_path.is_absolute():
            img_path = self.image_dir / img_path
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        label = torch.tensor(row['y_true'], dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'patient_id': row['patient_id'],
            'image_path': str(row['image_path']),
            'idx': idx
        }
    
    def get_patient_info(self):
        """Return patient-level information for this dataset."""
        return {
            'patient_ids': self.patient_ids,
            'patient_to_indices': self.patient_to_indices,
            'patient_to_label': self.patient_to_label,
            'n_patients': len(self.patient_ids),
            'n_samples': len(self.df)
        }

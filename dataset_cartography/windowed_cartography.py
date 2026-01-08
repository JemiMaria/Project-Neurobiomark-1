"""
Windowed Cartography Computation Module.

This module implements the windowed epoch strategy for computing
reliable confidence and correctness signals at image and patient levels.

The aggregation order is: Epoch → Image → Patient → Seed
"""

import os
import numpy as np
import pandas as pd

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    WINDOW_HALF_SIZE, OUTPUT_DIR,
    CARTOGRAPHY_PER_IMAGE_XLSX, CARTOGRAPHY_PER_PATIENT_XLSX
)


def sigmoid(x):
    """Compute sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_window_range(best_epoch, num_epochs, half_size=None):
    """
    Compute the epoch window range centered on best validation loss epoch.
    
    Args:
        best_epoch: 0-indexed epoch with best val_loss (t*)
        num_epochs: Total number of epochs trained
        half_size: Half-size of window (default from config)
        
    Returns:
        tuple: (start_epoch, end_epoch) - both inclusive, 0-indexed
    """
    if half_size is None:
        half_size = WINDOW_HALF_SIZE
    
    # Window: [t* - half_size, t* + half_size]
    start_epoch = max(0, best_epoch - half_size)
    end_epoch = min(num_epochs - 1, best_epoch + half_size)
    
    return start_epoch, end_epoch


def compute_image_level_metrics_single_seed(epoch_predictions, best_epoch):
    """
    Compute image-level cartography signals for a single seed/run.
    
    Step 1: For each image, compute confidence/correctness per epoch in window
    Step 2: Average across epochs in window to get image-level values
    
    Args:
        epoch_predictions: List of lists, where each inner list contains 
                          prediction dicts for one epoch
                          [{image_id, patient_id, y_true, y_pred_logit}, ...]
        best_epoch: 0-indexed epoch with best val_loss
        
    Returns:
        DataFrame with columns: image_id, patient_id, y_true, mean_confidence, 
                               mean_correctness, window_start, window_end, t_star
    """
    num_epochs = len(epoch_predictions)
    
    # Compute window range
    start_epoch, end_epoch = compute_window_range(best_epoch, num_epochs)
    window_size = end_epoch - start_epoch + 1
    
    print(f"  Window: epochs [{start_epoch}, {end_epoch}] (t*={best_epoch}, size={window_size})")
    
    # Build a mapping: image_id -> list of (epoch_idx, prediction_dict)
    image_epoch_data = {}
    
    for epoch_idx in range(start_epoch, end_epoch + 1):
        epoch_preds = epoch_predictions[epoch_idx]
        for pred in epoch_preds:
            img_id = pred['image_id']
            if img_id not in image_epoch_data:
                image_epoch_data[img_id] = {
                    'patient_id': pred['patient_id'],
                    'y_true': pred['y_true'],
                    'confidences': [],
                    'correctnesses': []
                }
            
            # Compute confidence for this epoch
            logit = pred['y_pred_logit']
            y_true = pred['y_true']
            prob = sigmoid(logit)
            
            # Confidence = probability assigned to true label
            if y_true == 1:
                confidence = prob
            else:
                confidence = 1.0 - prob
            
            # Correctness = 1 if prediction matches true label
            pred_label = 1 if prob >= 0.5 else 0
            correctness = 1 if pred_label == y_true else 0
            
            image_epoch_data[img_id]['confidences'].append(confidence)
            image_epoch_data[img_id]['correctnesses'].append(correctness)
    
    # Step 2: Average across epochs in window for each image
    results = []
    for img_id, data in image_epoch_data.items():
        mean_confidence = np.mean(data['confidences'])
        mean_correctness = np.mean(data['correctnesses'])
        
        results.append({
            'image_id': img_id,
            'patient_id': data['patient_id'],
            'y_true': data['y_true'],
            'mean_confidence': mean_confidence,
            'mean_correctness': mean_correctness,
            'window_start': start_epoch,
            'window_end': end_epoch,
            'window_size': window_size,
            't_star': best_epoch
        })
    
    return pd.DataFrame(results)


def compute_patient_level_metrics_single_seed(image_df):
    """
    Aggregate image-level metrics to patient-level for a single seed.
    
    For each patient, average the already-epoch-averaged image values.
    Each patient gets equal weight, regardless of number of images.
    
    Args:
        image_df: DataFrame from compute_image_level_metrics_single_seed
        
    Returns:
        DataFrame with columns: patient_id, patient_mean_confidence, 
                               patient_mean_correctness, num_samples_in_patient,
                               window_size, t_star
    """
    # Group by patient_id and compute mean of image-level values
    patient_results = []
    
    for patient_id, group in image_df.groupby('patient_id'):
        patient_mean_confidence = group['mean_confidence'].mean()
        patient_mean_correctness = group['mean_correctness'].mean()
        num_samples = len(group)
        window_size = group['window_size'].iloc[0]
        t_star = group['t_star'].iloc[0]
        
        patient_results.append({
            'patient_id': patient_id,
            'patient_mean_confidence': patient_mean_confidence,
            'patient_mean_correctness': patient_mean_correctness,
            'num_samples_in_patient': num_samples,
            'window_size': window_size,
            't_star': t_star
        })
    
    return pd.DataFrame(patient_results)


def compute_windowed_cartography(all_epoch_predictions, best_epochs, metadata):
    """
    Compute windowed cartography metrics across all seeds.
    
    Aggregation order: Epoch → Image → Patient → Seed
    
    Args:
        all_epoch_predictions: List of per-run epoch predictions
                              [run1_predictions, run2_predictions, ...]
                              where each run has [epoch1_preds, epoch2_preds, ...]
        best_epochs: List of best epochs (0-indexed) per run
        metadata: DataFrame with image metadata (for reference)
        
    Returns:
        tuple: (image_df, patient_df)
            - image_df: Per-image table averaged across seeds
            - patient_df: Per-patient table averaged across seeds
    """
    num_seeds = len(all_epoch_predictions)
    
    print(f"\n{'='*60}")
    print("COMPUTING WINDOWED CARTOGRAPHY METRICS")
    print(f"{'='*60}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Best epochs per seed (0-indexed): {best_epochs}")
    
    # Store per-seed results
    seed_image_dfs = []
    seed_patient_dfs = []
    
    for seed_idx in range(num_seeds):
        print(f"\n--- Seed {seed_idx + 1} ---")
        epoch_predictions = all_epoch_predictions[seed_idx]
        best_epoch = best_epochs[seed_idx]
        
        # Step 1 & 2: Compute image-level metrics
        image_df = compute_image_level_metrics_single_seed(epoch_predictions, best_epoch)
        seed_image_dfs.append(image_df)
        print(f"  Image-level: {len(image_df)} images processed")
        
        # Step 3: Compute patient-level metrics
        patient_df = compute_patient_level_metrics_single_seed(image_df)
        seed_patient_dfs.append(patient_df)
        print(f"  Patient-level: {len(patient_df)} patients processed")
    
    # Step 4: Cross-seed aggregation
    print(f"\n--- Cross-Seed Aggregation ---")
    
    # Aggregate image-level across seeds
    final_image_df = aggregate_across_seeds_image_level(seed_image_dfs)
    print(f"Final image-level: {len(final_image_df)} images")
    
    # Aggregate patient-level across seeds
    final_patient_df = aggregate_across_seeds_patient_level(seed_patient_dfs)
    print(f"Final patient-level: {len(final_patient_df)} patients")
    
    return final_image_df, final_patient_df


def aggregate_across_seeds_image_level(seed_image_dfs):
    """
    Aggregate image-level metrics across all seeds.
    
    For each image, average the seed-specific confidence and correctness.
    
    Args:
        seed_image_dfs: List of DataFrames, one per seed
        
    Returns:
        DataFrame with aggregated image-level metrics
    """
    if len(seed_image_dfs) == 1:
        # Only one seed, return as-is
        return seed_image_dfs[0]
    
    # Combine all seed DataFrames
    combined = pd.concat(seed_image_dfs, ignore_index=True)
    
    # Aggregate by image_id
    aggregated = combined.groupby('image_id').agg({
        'patient_id': 'first',  # Same for all seeds
        'y_true': 'first',       # Same for all seeds
        'mean_confidence': 'mean',   # Average across seeds
        'mean_correctness': 'mean',  # Average across seeds
        'window_size': lambda x: list(x),  # Keep track of window sizes
        't_star': lambda x: list(x)        # Keep track of t* values
    }).reset_index()
    
    # Format window_size and t_star as strings showing all seeds
    aggregated['window_sizes'] = aggregated['window_size'].apply(lambda x: ','.join(map(str, x)))
    aggregated['t_stars'] = aggregated['t_star'].apply(lambda x: ','.join(map(str, x)))
    
    # Drop intermediate columns
    aggregated = aggregated.drop(columns=['window_size', 't_star'])
    aggregated = aggregated.rename(columns={
        'window_sizes': 'window_sizes_per_seed',
        't_stars': 't_stars_per_seed'
    })
    
    return aggregated


def aggregate_across_seeds_patient_level(seed_patient_dfs):
    """
    Aggregate patient-level metrics across all seeds.
    
    For each patient, average the seed-specific patient confidence and correctness.
    
    Args:
        seed_patient_dfs: List of DataFrames, one per seed
        
    Returns:
        DataFrame with aggregated patient-level metrics
    """
    if len(seed_patient_dfs) == 1:
        # Only one seed, return as-is
        return seed_patient_dfs[0]
    
    # Combine all seed DataFrames
    combined = pd.concat(seed_patient_dfs, ignore_index=True)
    
    # Aggregate by patient_id
    aggregated = combined.groupby('patient_id').agg({
        'patient_mean_confidence': 'mean',    # Average across seeds
        'patient_mean_correctness': 'mean',   # Average across seeds
        'num_samples_in_patient': 'first',    # Same for all seeds
        'window_size': lambda x: list(x),     # Keep track of window sizes
        't_star': lambda x: list(x)           # Keep track of t* values
    }).reset_index()
    
    # Format window_size and t_star as strings showing all seeds
    aggregated['window_sizes'] = aggregated['window_size'].apply(lambda x: ','.join(map(str, x)))
    aggregated['t_stars'] = aggregated['t_star'].apply(lambda x: ','.join(map(str, x)))
    
    # Drop intermediate columns
    aggregated = aggregated.drop(columns=['window_size', 't_star'])
    aggregated = aggregated.rename(columns={
        'window_sizes': 'window_sizes_per_seed',
        't_stars': 't_stars_per_seed'
    })
    
    return aggregated


def save_windowed_cartography_results(image_df, patient_df, output_dir=None):
    """
    Save windowed cartography results to Excel files.
    
    Args:
        image_df: Per-image DataFrame
        patient_df: Per-patient DataFrame
        output_dir: Output directory (default from config)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-image table
    image_path = output_dir / CARTOGRAPHY_PER_IMAGE_XLSX
    
    # Round numeric columns to 4 decimals
    numeric_cols_image = ['mean_confidence', 'mean_correctness']
    image_df_formatted = image_df.copy()
    for col in numeric_cols_image:
        if col in image_df_formatted.columns:
            image_df_formatted[col] = image_df_formatted[col].round(4)
    
    image_df_formatted.to_excel(image_path, index=False, engine='openpyxl')
    print(f"Per-image cartography saved to: {image_path}")
    
    # Save per-patient table
    patient_path = output_dir / CARTOGRAPHY_PER_PATIENT_XLSX
    
    # Round numeric columns to 4 decimals
    numeric_cols_patient = ['patient_mean_confidence', 'patient_mean_correctness']
    patient_df_formatted = patient_df.copy()
    for col in numeric_cols_patient:
        if col in patient_df_formatted.columns:
            patient_df_formatted[col] = patient_df_formatted[col].round(4)
    
    patient_df_formatted.to_excel(patient_path, index=False, engine='openpyxl')
    print(f"Per-patient cartography saved to: {patient_path}")
    
    return image_path, patient_path


def validate_cartography_results(image_df, patient_df, expected_patients=15):
    """
    Validate cartography results for sanity checks.
    
    Args:
        image_df: Per-image DataFrame
        patient_df: Per-patient DataFrame
        expected_patients: Expected number of patients
        
    Returns:
        dict: Validation results with pass/fail status
    """
    print(f"\n{'='*60}")
    print("VALIDATION CHECKLIST")
    print(f"{'='*60}")
    
    results = {}
    
    # Check 1: Per-image table row count
    num_images = len(image_df)
    results['image_count'] = num_images
    print(f"[✓] Per-image table: {num_images} rows")
    
    # Check 2: Per-patient table row count
    num_patients = len(patient_df)
    patient_check = num_patients == expected_patients
    status = "✓" if patient_check else "✗"
    results['patient_count'] = num_patients
    results['patient_count_ok'] = patient_check
    print(f"[{status}] Per-patient table: {num_patients} rows (expected: {expected_patients})")
    
    # Check 3: Confidence values in [0, 1]
    conf_min = image_df['mean_confidence'].min()
    conf_max = image_df['mean_confidence'].max()
    conf_check = (conf_min >= 0) and (conf_max <= 1)
    status = "✓" if conf_check else "✗"
    results['confidence_range_ok'] = conf_check
    print(f"[{status}] Confidence values: [{conf_min:.4f}, {conf_max:.4f}] (expected: [0, 1])")
    
    # Check 4: Correctness values in [0, 1]
    corr_min = image_df['mean_correctness'].min()
    corr_max = image_df['mean_correctness'].max()
    corr_check = (corr_min >= 0) and (corr_max <= 1)
    status = "✓" if corr_check else "✗"
    results['correctness_range_ok'] = corr_check
    print(f"[{status}] Correctness values: [{corr_min:.4f}, {corr_max:.4f}] (expected: [0, 1])")
    
    # Check 5: No missing patient_id
    missing_patient = image_df['patient_id'].isna().sum()
    missing_check = missing_patient == 0
    status = "✓" if missing_check else "✗"
    results['no_missing_patient_id'] = missing_check
    print(f"[{status}] Missing patient_id values: {missing_patient}")
    
    # Summary statistics
    print(f"\n--- Summary Statistics ---")
    print(f"Mean confidence (images): {image_df['mean_confidence'].mean():.4f}")
    print(f"Mean correctness (images): {image_df['mean_correctness'].mean():.4f}")
    print(f"Mean confidence (patients): {patient_df['patient_mean_confidence'].mean():.4f}")
    print(f"Mean correctness (patients): {patient_df['patient_mean_correctness'].mean():.4f}")
    
    return results


def print_cartography_summary(image_df, patient_df):
    """
    Print a summary of windowed cartography results.
    
    Args:
        image_df: Per-image DataFrame
        patient_df: Per-patient DataFrame
    """
    print(f"\n{'='*60}")
    print("WINDOWED CARTOGRAPHY SUMMARY")
    print(f"{'='*60}")
    
    # Image-level summary
    print("\n--- Image-Level Summary ---")
    print(f"Total images: {len(image_df)}")
    print(f"  Class 0 (Control): {(image_df['y_true'] == 0).sum()}")
    print(f"  Class 1 (Case): {(image_df['y_true'] == 1).sum()}")
    print(f"\nConfidence statistics:")
    print(f"  Mean: {image_df['mean_confidence'].mean():.4f}")
    print(f"  Std:  {image_df['mean_confidence'].std():.4f}")
    print(f"  Min:  {image_df['mean_confidence'].min():.4f}")
    print(f"  Max:  {image_df['mean_confidence'].max():.4f}")
    print(f"\nCorrectness statistics:")
    print(f"  Mean: {image_df['mean_correctness'].mean():.4f}")
    print(f"  Std:  {image_df['mean_correctness'].std():.4f}")
    
    # Patient-level summary
    print("\n--- Patient-Level Summary ---")
    print(f"Total patients: {len(patient_df)}")
    print(f"\nConfidence statistics:")
    print(f"  Mean: {patient_df['patient_mean_confidence'].mean():.4f}")
    print(f"  Std:  {patient_df['patient_mean_confidence'].std():.4f}")
    print(f"  Min:  {patient_df['patient_mean_confidence'].min():.4f}")
    print(f"  Max:  {patient_df['patient_mean_confidence'].max():.4f}")
    print(f"\nCorrectness statistics:")
    print(f"  Mean: {patient_df['patient_mean_correctness'].mean():.4f}")
    print(f"  Std:  {patient_df['patient_mean_correctness'].std():.4f}")
    
    # Per-patient details
    print("\n--- Per-Patient Details ---")
    sorted_patients = patient_df.sort_values('patient_mean_correctness', ascending=False)
    print(sorted_patients[['patient_id', 'patient_mean_confidence', 
                           'patient_mean_correctness', 'num_samples_in_patient']].to_string(index=False))


if __name__ == "__main__":
    print("Windowed Cartography Module")
    print("This module should be called from run_cartography.py")

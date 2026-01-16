"""
Windowed Cartography for ALS vs ALS-FTD.

Computes cartography metrics using a window around the best validation epoch:
1. Find t* = argmin(val_loss)
2. Window = [t*-2, t*+2]
3. Compute confidence, correctness, variability within window
4. Aggregate: epoch → image → patient → seed

Metrics:
- confidence = P(y_pred = y_true) = prob if y=1 else 1-prob
- correctness = 1 if prediction correct else 0
- variability = std(prob) across window epochs
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    WINDOW_HALF_SIZE, CARTOGRAPHY_DIR, RANDOM_SEEDS
)


def compute_windowed_cartography_single_seed(epoch_predictions, history, fold_id, seed):
    """
    Compute windowed cartography metrics for a single fold-seed.
    
    Args:
        epoch_predictions: List of per-epoch prediction dicts
        history: Training history with val_loss
        fold_id: Fold ID
        seed: Random seed
        
    Returns:
        pd.DataFrame: Per-image metrics for this fold-seed
    """
    # Convert to DataFrame
    pred_df = pd.DataFrame(epoch_predictions)
    
    # Find best epoch (t*)
    best_epoch = np.argmin(history['val_loss'])
    max_epoch = max(history['epoch'])
    
    # Define window [t*-2, t*+2]
    window_start = max(0, best_epoch - WINDOW_HALF_SIZE)
    window_end = min(max_epoch, best_epoch + WINDOW_HALF_SIZE)
    window_epochs = list(range(window_start, window_end + 1))
    
    # Filter to window epochs
    window_df = pred_df[pred_df['epoch'].isin(window_epochs)].copy()
    
    if len(window_df) == 0:
        print(f"    Warning: No predictions in window for fold {fold_id}, seed {seed}")
        return pd.DataFrame()
    
    # Compute per-epoch metrics for each image
    image_metrics = []
    
    for image_path in window_df['image_path'].unique():
        img_data = window_df[window_df['image_path'] == image_path]
        
        y_true = img_data['y_true'].iloc[0]
        patient_id = img_data['patient_id'].iloc[0]
        
        # Compute confidence for each epoch
        # confidence = prob if y=1 else (1-prob)
        confidences = []
        correctnesses = []
        probs = []
        
        for _, row in img_data.iterrows():
            prob = row['prob']
            probs.append(prob)
            
            if y_true == 1:
                confidence = prob
            else:
                confidence = 1 - prob
            confidences.append(confidence)
            
            # correctness = 1 if prediction correct else 0
            pred_label = 1 if prob >= 0.5 else 0
            correctness = 1 if pred_label == y_true else 0
            correctnesses.append(correctness)
        
        # Aggregate within window
        mean_confidence_window = np.mean(confidences)
        mean_correctness_window = np.mean(correctnesses)
        std_prob_window = np.std(probs)
        mean_prob_window = np.mean(probs)
        
        image_metrics.append({
            'fold_id': fold_id,
            'seed': seed,
            'image_path': image_path,
            'patient_id': patient_id,
            'y_true': y_true,
            'mean_confidence_window': mean_confidence_window,
            'mean_correctness_window': mean_correctness_window,
            'std_prob_window': std_prob_window,
            'mean_prob_window': mean_prob_window,
            'best_epoch': best_epoch,
            'window_start': window_start,
            'window_end': window_end,
            'n_window_epochs': len(window_epochs)
        })
    
    return pd.DataFrame(image_metrics)


def aggregate_to_patient_level_single_seed(image_df):
    """
    Aggregate image-level metrics to patient-level for a single seed.
    
    For each patient, average the image-level values.
    
    Args:
        image_df: DataFrame with per-image metrics
        
    Returns:
        pd.DataFrame: Per-patient metrics
    """
    if len(image_df) == 0:
        return pd.DataFrame()
    
    patient_metrics = []
    
    for patient_id in image_df['patient_id'].unique():
        patient_data = image_df[image_df['patient_id'] == patient_id]
        
        patient_metrics.append({
            'fold_id': patient_data['fold_id'].iloc[0],
            'seed': patient_data['seed'].iloc[0],
            'patient_id': patient_id,
            'y_true': patient_data['y_true'].iloc[0],
            'patient_mean_confidence': patient_data['mean_confidence_window'].mean(),
            'patient_mean_correctness': patient_data['mean_correctness_window'].mean(),
            'patient_std_prob': patient_data['std_prob_window'].mean(),
            'patient_mean_prob': patient_data['mean_prob_window'].mean(),
            'num_images': len(patient_data)
        })
    
    return pd.DataFrame(patient_metrics)


def aggregate_across_seeds(patient_seed_df):
    """
    Aggregate patient metrics across seeds.
    
    For each patient (across LOPO - held out in their fold):
    - final_mean_confidence = mean(patient_mean_confidence across seeds)
    - final_mean_correctness = mean(patient_mean_correctness across seeds)
    - final_std_prob = mean(patient_std_prob across seeds)
    - final_std_correctness = std(patient_mean_correctness across seeds)
    - flip_rate = fraction of seeds with different prediction
    
    Args:
        patient_seed_df: DataFrame with per-patient-seed metrics
        
    Returns:
        pd.DataFrame: Final patient-level metrics
    """
    if len(patient_seed_df) == 0:
        return pd.DataFrame()
    
    final_metrics = []
    
    for patient_id in patient_seed_df['patient_id'].unique():
        patient_data = patient_seed_df[patient_seed_df['patient_id'] == patient_id]
        
        # Compute flip rate
        # A flip is when prediction differs from majority prediction
        probs = patient_data['patient_mean_prob'].values
        preds = (probs >= 0.5).astype(int)
        majority_pred = 1 if np.mean(preds) >= 0.5 else 0
        flips = np.sum(preds != majority_pred)
        flip_rate = flips / len(preds)
        
        final_metrics.append({
            'patient_id': patient_id,
            'y_true': patient_data['y_true'].iloc[0],
            'final_mean_confidence': patient_data['patient_mean_confidence'].mean(),
            'final_mean_correctness': patient_data['patient_mean_correctness'].mean(),
            'final_std_prob': patient_data['patient_std_prob'].mean(),
            'final_std_correctness': patient_data['patient_mean_correctness'].std(),
            'final_mean_prob': patient_data['patient_mean_prob'].mean(),
            'flip_rate': flip_rate,
            'num_images': int(patient_data['num_images'].iloc[0]),
            'num_seeds': len(patient_data)
        })
    
    return pd.DataFrame(final_metrics)


def save_cartography_tables(image_df, patient_final_df, output_dir=None):
    """
    Save cartography tables to Excel files.
    
    Args:
        image_df: Per-image metrics DataFrame
        patient_final_df: Final patient-level metrics DataFrame
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = CARTOGRAPHY_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-image table
    image_path = output_dir / "cartography_per_image.xlsx"
    image_df.to_excel(image_path, index=False, engine='openpyxl')
    print(f"  ✓ Saved per-image cartography: {image_path}")
    
    # Save final patient table
    patient_path = output_dir / "cartography_per_patient_final.xlsx"
    patient_final_df.to_excel(patient_path, index=False, engine='openpyxl')
    print(f"  ✓ Saved per-patient cartography: {patient_path}")


def run_cartography_pipeline(all_fold_seed_results):
    """
    Run complete cartography pipeline across all folds and seeds.
    
    Args:
        all_fold_seed_results: List of dicts with 'fold_id', 'seed', 'epoch_predictions', 'history'
        
    Returns:
        tuple: (image_df, patient_final_df)
    """
    print(f"\n{'='*60}")
    print("COMPUTING WINDOWED CARTOGRAPHY")
    print(f"{'='*60}")
    
    all_image_metrics = []
    all_patient_seed_metrics = []
    
    for result in all_fold_seed_results:
        fold_id = result['fold_id']
        seed = result['seed']
        epoch_predictions = result['epoch_predictions']
        history = result['history']
        
        # Compute per-image metrics
        image_df = compute_windowed_cartography_single_seed(
            epoch_predictions, history, fold_id, seed
        )
        
        if len(image_df) > 0:
            all_image_metrics.append(image_df)
            
            # Aggregate to patient level
            patient_df = aggregate_to_patient_level_single_seed(image_df)
            if len(patient_df) > 0:
                all_patient_seed_metrics.append(patient_df)
    
    # Combine all image metrics
    if all_image_metrics:
        combined_image_df = pd.concat(all_image_metrics, ignore_index=True)
    else:
        combined_image_df = pd.DataFrame()
    
    # Combine all patient-seed metrics
    if all_patient_seed_metrics:
        combined_patient_seed_df = pd.concat(all_patient_seed_metrics, ignore_index=True)
    else:
        combined_patient_seed_df = pd.DataFrame()
    
    # Aggregate across seeds
    print("\n  Aggregating across seeds...")
    patient_final_df = aggregate_across_seeds(combined_patient_seed_df)
    
    # Save tables
    print("\n  Saving cartography tables...")
    save_cartography_tables(combined_image_df, patient_final_df)
    
    print(f"\n  Summary:")
    print(f"    - Total images: {len(combined_image_df)}")
    print(f"    - Total patients: {len(patient_final_df)}")
    
    return combined_image_df, patient_final_df

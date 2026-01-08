"""
LOPO Runner - Main orchestration for Leave-One-Patient-Out evaluation.

This module:
- Implements the LOPO splitting loop
- Wraps existing training pipeline for each fold
- Applies windowed cartography per fold
- Saves per-image and per-patient results
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import copy
import time
from datetime import timedelta

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_cartography.config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, LABEL_SMOOTHING,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
    RANDOM_SEEDS, OUTPUT_DIR, WINDOW_HALF_SIZE
)
from dataset_cartography.model import create_model, get_device
from dataset_cartography.data_loader import (
    load_metadata, BrainTissueDataset, get_image_transforms,
    create_weighted_sampler
)
from dataset_cartography.trainer import (
    EarlyStopping, train_one_epoch, validate,
    compute_classification_metrics
)

# LOPO output directory
LOPO_DIR = OUTPUT_DIR / "lopo"
LOPO_ANALYSIS_DIR = LOPO_DIR / "analysis"


def ensure_lopo_directories():
    """Create LOPO output directories."""
    LOPO_DIR.mkdir(parents=True, exist_ok=True)
    LOPO_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return LOPO_DIR, LOPO_ANALYSIS_DIR


def sigmoid(x):
    """Compute sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def get_lopo_splits(metadata):
    """
    Generate LOPO splits - one fold per patient.
    
    Args:
        metadata: DataFrame with patient_id column
        
    Yields:
        tuple: (fold_idx, held_out_patient, train_df, test_df)
    """
    unique_patients = sorted(metadata['patient_id'].unique())
    
    for fold_idx, held_out_patient in enumerate(unique_patients):
        train_df = metadata[metadata['patient_id'] != held_out_patient].copy()
        test_df = metadata[metadata['patient_id'] == held_out_patient].copy()
        
        yield fold_idx, held_out_patient, train_df, test_df


def create_fold_dataloaders(train_df, test_df, batch_size=8):
    """
    Create dataloaders for a single LOPO fold.
    
    Args:
        train_df: Training metadata (all patients except held-out)
        test_df: Test metadata (held-out patient only)
        batch_size: Batch size for dataloaders
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    transform = get_image_transforms()
    
    # Create datasets
    train_dataset = BrainTissueDataset(train_df, transform=transform)
    test_dataset = BrainTissueDataset(test_df, transform=transform)
    
    # Create weighted sampler for training (class balance)
    train_labels = train_df['label'].values
    sampler = create_weighted_sampler(train_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader


def compute_epoch_predictions_lopo(model, dataloader, metadata, device):
    """
    Compute predictions for held-out patient images.
    
    Args:
        model: Trained model
        dataloader: DataLoader for held-out patient
        metadata: DataFrame with patient_id mapping
        device: Device to run on
        
    Returns:
        list: List of prediction dicts with image_id, patient_id, y_true, y_pred_logit, prob
    """
    model.eval()
    predictions = []
    
    img_to_patient = dict(zip(metadata['image_no'], metadata['patient_id']))
    
    with torch.no_grad():
        for images, labels, image_nos in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            for i in range(len(labels)):
                img_no = int(image_nos[i])
                true_label = int(labels[i])
                logit = float(outputs[i])
                prob = float(sigmoid(logit))
                patient_id = img_to_patient.get(img_no, "Unknown")
                
                predictions.append({
                    'image_id': img_no,
                    'patient_id': patient_id,
                    'y_true': true_label,
                    'y_pred_logit': logit,
                    'prob': prob
                })
    
    return predictions


def train_single_fold_seed(train_loader, test_loader, test_df, fold_idx, 
                           held_out_patient, seed, device):
    """
    Train model for a single LOPO fold and seed.
    
    Args:
        train_loader: Training dataloader
        test_loader: Test dataloader (held-out patient)
        test_df: Test metadata
        fold_idx: Fold index
        held_out_patient: Patient ID being held out
        seed: Random seed
        device: Device to run on
        
    Returns:
        tuple: (epoch_predictions, training_log, best_epoch)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create fresh model
    model = create_model(pretrained=True)
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Storage
    epoch_predictions = []  # Store predictions for each epoch
    training_log = []
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc, _, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on held-out patient
        test_loss, test_acc, test_labels, test_probs, test_preds = validate(
            model, test_loader, criterion, device
        )
        
        # Compute metrics
        test_metrics = compute_classification_metrics(test_labels, test_probs, test_preds)
        
        # Log
        training_log.append({
            'fold': fold_idx,
            'held_out_patient': held_out_patient,
            'seed': seed,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_metrics['auc'],
            'test_f1': test_metrics['f1']
        })
        
        # Scheduler step
        scheduler.step(test_loss)
        
        # Early stopping check
        is_best = early_stopping(test_loss)
        if is_best:
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Store predictions for this epoch
        preds = compute_epoch_predictions_lopo(model, test_loader, test_df, device)
        epoch_predictions.append(preds)
        
        if early_stopping.should_stop:
            break
    
    return epoch_predictions, training_log, best_epoch


def compute_windowed_metrics_lopo(epoch_predictions, best_epoch, half_size=None):
    """
    Compute windowed cartography metrics for LOPO fold.
    
    Args:
        epoch_predictions: List of prediction lists per epoch
        best_epoch: Best epoch (0-indexed)
        half_size: Window half-size
        
    Returns:
        DataFrame with per-image windowed metrics
    """
    if half_size is None:
        half_size = WINDOW_HALF_SIZE
    
    num_epochs = len(epoch_predictions)
    start_epoch = max(0, best_epoch - half_size)
    end_epoch = min(num_epochs - 1, best_epoch + half_size)
    
    # Collect per-image data across window
    image_data = {}
    
    for epoch_idx in range(start_epoch, end_epoch + 1):
        for pred in epoch_predictions[epoch_idx]:
            img_id = pred['image_id']
            if img_id not in image_data:
                image_data[img_id] = {
                    'patient_id': pred['patient_id'],
                    'y_true': pred['y_true'],
                    'probs': [],
                    'confidences': [],
                    'correctnesses': []
                }
            
            prob = pred['prob']
            y_true = pred['y_true']
            
            # Confidence = prob assigned to true label
            confidence = prob if y_true == 1 else 1 - prob
            
            # Correctness
            pred_label = 1 if prob >= 0.5 else 0
            correctness = 1 if pred_label == y_true else 0
            
            image_data[img_id]['probs'].append(prob)
            image_data[img_id]['confidences'].append(confidence)
            image_data[img_id]['correctnesses'].append(correctness)
    
    # Aggregate per image
    results = []
    for img_id, data in image_data.items():
        results.append({
            'image_id': img_id,
            'patient_id': data['patient_id'],
            'y_true': data['y_true'],
            'mean_prob_window': np.mean(data['probs']),
            'std_prob_window': np.std(data['probs']),
            'mean_confidence_window': np.mean(data['confidences']),
            'mean_correctness_window': np.mean(data['correctnesses']),
            'window_start': start_epoch,
            'window_end': end_epoch,
            't_star': best_epoch
        })
    
    return pd.DataFrame(results)


def run_lopo_evaluation(metadata=None, seeds=None, device=None):
    """
    Run complete LOPO evaluation.
    
    Args:
        metadata: DataFrame with image metadata (loads if None)
        seeds: List of random seeds
        device: Device to run on
        
    Returns:
        dict: All LOPO results
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print("LEAVE-ONE-PATIENT-OUT (LOPO) EVALUATION")
    print("="*70)
    
    # Setup
    ensure_lopo_directories()
    
    if metadata is None:
        metadata = load_metadata()
    if seeds is None:
        seeds = RANDOM_SEEDS
    if device is None:
        device = get_device()
    
    unique_patients = sorted(metadata['patient_id'].unique())
    num_folds = len(unique_patients)
    num_seeds = len(seeds)
    
    print(f"Total patients: {num_folds}")
    print(f"Total folds: {num_folds}")
    print(f"Seeds per fold: {num_seeds}")
    print(f"Total training runs: {num_folds * num_seeds}")
    
    # Storage for all results
    all_image_results = []      # Per-image windowed metrics
    all_training_logs = []      # Training logs
    
    # LOPO Loop
    for fold_idx, held_out_patient, train_df, test_df in get_lopo_splits(metadata):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{num_folds}: Held-out patient = {held_out_patient}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_df)} images from {train_df['patient_id'].nunique()} patients")
        print(f"  Test:  {len(test_df)} images from patient {held_out_patient}")
        
        # Create dataloaders for this fold
        train_loader, test_loader = create_fold_dataloaders(train_df, test_df)
        
        # Train with each seed
        fold_image_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n  --- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) ---")
            
            # Train single fold/seed
            epoch_preds, train_log, best_epoch = train_single_fold_seed(
                train_loader, test_loader, test_df,
                fold_idx, held_out_patient, seed, device
            )
            
            all_training_logs.extend(train_log)
            
            # Compute windowed metrics
            image_df = compute_windowed_metrics_lopo(epoch_preds, best_epoch)
            image_df['fold_patient_id'] = held_out_patient
            image_df['seed'] = seed
            image_df['fold_idx'] = fold_idx
            
            fold_image_results.append(image_df)
            
            print(f"    Best epoch: {best_epoch + 1}, Images: {len(image_df)}")
        
        # Combine seed results for this fold
        fold_combined = pd.concat(fold_image_results, ignore_index=True)
        all_image_results.append(fold_combined)
    
    # Combine all folds
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)
    
    # A) Per-image table (all folds, all seeds)
    lopo_per_image = pd.concat(all_image_results, ignore_index=True)
    
    # B) Per-patient-seed table
    lopo_per_patient_seed = lopo_per_image.groupby(
        ['fold_patient_id', 'seed', 'patient_id']
    ).agg({
        'mean_confidence_window': 'mean',
        'mean_correctness_window': 'mean',
        'mean_prob_window': 'mean',
        'std_prob_window': 'mean',
        'y_true': 'first',
        'image_id': 'count'
    }).reset_index()
    lopo_per_patient_seed = lopo_per_patient_seed.rename(columns={
        'mean_confidence_window': 'patient_mean_confidence',
        'mean_correctness_window': 'patient_mean_correctness',
        'mean_prob_window': 'patient_mean_prob',
        'std_prob_window': 'patient_std_prob',
        'image_id': 'n_images'
    })
    
    # C) Per-patient final (aggregate across seeds)
    lopo_per_patient_final = lopo_per_patient_seed.groupby('fold_patient_id').agg({
        'patient_mean_confidence': ['mean', 'std'],
        'patient_mean_correctness': ['mean', 'std'],
        'patient_mean_prob': ['mean', 'std'],
        'y_true': 'first',
        'n_images': 'sum'
    }).reset_index()
    
    # Flatten columns
    lopo_per_patient_final.columns = [
        'fold_patient_id',
        'mean_confidence_across_seeds', 'std_confidence_across_seeds',
        'mean_correctness_across_seeds', 'std_correctness_across_seeds',
        'mean_prob_across_seeds', 'std_prob_across_seeds',
        'y_true', 'n_images_total'
    ]
    
    # Save results
    print("\nSaving LOPO results...")
    
    lopo_per_image.round(4).to_excel(LOPO_DIR / "lopo_per_image.xlsx", index=False)
    print(f"  ✓ {LOPO_DIR / 'lopo_per_image.xlsx'}")
    
    lopo_per_patient_seed.round(4).to_excel(LOPO_DIR / "lopo_per_patient_seed.xlsx", index=False)
    print(f"  ✓ {LOPO_DIR / 'lopo_per_patient_seed.xlsx'}")
    
    lopo_per_patient_final.round(4).to_excel(LOPO_DIR / "lopo_per_patient_final.xlsx", index=False)
    print(f"  ✓ {LOPO_DIR / 'lopo_per_patient_final.xlsx'}")
    
    training_logs_df = pd.DataFrame(all_training_logs)
    training_logs_df.to_csv(LOPO_DIR / "lopo_training_logs.csv", index=False)
    print(f"  ✓ {LOPO_DIR / 'lopo_training_logs.csv'}")
    
    # Compute clinical metrics
    from .lopo_metrics import compute_clinical_metrics
    clinical_metrics = compute_clinical_metrics(lopo_per_image, lopo_per_patient_final)
    clinical_metrics.to_csv(LOPO_DIR / "lopo_clinical_metrics.csv", index=False)
    print(f"  ✓ {LOPO_DIR / 'lopo_clinical_metrics.csv'}")
    
    # Create visualizations
    from .lopo_visualize import create_lopo_visualizations
    create_lopo_visualizations(lopo_per_patient_final, lopo_per_patient_seed, 
                               lopo_per_image, clinical_metrics, LOPO_ANALYSIS_DIR)
    
    # Summary
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    print("\n" + "="*60)
    print("LOPO EVALUATION COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed_str}")
    print(f"Results saved to: {LOPO_DIR}")
    
    # Print summary statistics
    print("\n--- Patient-Level Summary ---")
    print(f"Mean correctness: {lopo_per_patient_final['mean_correctness_across_seeds'].mean():.4f}")
    print(f"Mean confidence: {lopo_per_patient_final['mean_confidence_across_seeds'].mean():.4f}")
    
    return {
        'per_image': lopo_per_image,
        'per_patient_seed': lopo_per_patient_seed,
        'per_patient_final': lopo_per_patient_final,
        'training_logs': training_logs_df,
        'clinical_metrics': clinical_metrics
    }


if __name__ == "__main__":
    results = run_lopo_evaluation()

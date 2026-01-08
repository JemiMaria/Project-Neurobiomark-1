"""
Training utilities for Dataset Cartography.

This module handles:
- Training loop with early stopping
- Epoch-wise tracking of predictions for windowed cartography
- Saving training logs and best model weights
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, 
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
    MODELS_DIR, LOGS_DIR, WINDOW_HALF_SIZE
)
from dataset_cartography.model import create_model, get_device, save_model


class EarlyStopping:
    """
    Early stopping to halt training when validation loss doesn't improve.
    """
    
    def __init__(self, patience=5, min_delta=0):
        """
        Initialize early stopping tracker.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            bool: True if this is the best model so far
        """
        if self.best_loss is None:
            # First epoch
            self.best_loss = val_loss
            return True
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def compute_confidence_and_correctness(model, dataloader, device):
    """
    Compute confidence and correctness for all samples in dataloader.
    
    Confidence: probability assigned to the TRUE label
    Correctness: 1 if prediction matches true label, 0 otherwise
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device to run on
        
    Returns:
        dict: {image_no: {'confidence': float, 'correctness': int}}
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for images, labels, image_nos in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            # Get model predictions (logits)
            outputs = model(images).squeeze()
            
            # Convert to probabilities
            probs = torch.sigmoid(outputs)
            
            # Process each sample in batch
            for i in range(len(labels)):
                img_no = int(image_nos[i])
                true_label = int(labels[i])
                prob_positive = float(probs[i]) if probs.dim() > 0 else float(probs)
                
                # Confidence: probability of the TRUE label
                if true_label == 1:
                    confidence = prob_positive
                else:
                    confidence = 1 - prob_positive
                
                # Correctness: 1 if prediction matches true label
                predicted = 1 if prob_positive >= 0.5 else 0
                correctness = 1 if predicted == true_label else 0
                
                results[img_no] = {
                    'confidence': confidence,
                    'correctness': correctness
                }
    
    return results


def compute_epoch_predictions(model, dataloader, metadata, device):
    """
    Compute detailed predictions for all samples in dataloader.
    
    This function stores raw logits and additional metadata needed for
    windowed cartography analysis at image and patient levels.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        metadata: DataFrame with patient_id mapping
        device: Device to run on
        
    Returns:
        list: List of dicts, each containing:
              {image_id, patient_id, y_true, y_pred_logit}
    """
    model.eval()
    predictions = []
    
    # Create mapping from image_no to patient_id
    img_to_patient = dict(zip(metadata['image_no'], metadata['patient_id']))
    
    with torch.no_grad():
        for images, labels, image_nos in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            # Get model predictions (logits)
            outputs = model(images).squeeze()
            
            # Handle single sample case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            # Process each sample in batch
            for i in range(len(labels)):
                img_no = int(image_nos[i])
                true_label = int(labels[i])
                logit = float(outputs[i])
                patient_id = img_to_patient.get(img_no, "Unknown")
                
                predictions.append({
                    'image_id': img_no,
                    'patient_id': patient_id,
                    'y_true': true_label,
                    'y_pred_logit': logit
                })
    
    return predictions


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch with label smoothing.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy, all_labels, all_probs, all_preds)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Collect all predictions for metrics calculation
    all_labels = []
    all_probs = []
    all_preds = []
    
    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        
        # Apply label smoothing: smooth labels away from 0 and 1
        # This reduces overconfidence and helps prevent overfitting
        smoothed_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).squeeze()
        
        # Compute loss with smoothed labels
        loss = criterion(outputs, smoothed_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Get probabilities and predictions (use original labels for metrics)
        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()
        
        # Track statistics (use original labels for accuracy)
        running_loss += loss.item() * images.size(0)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Collect for metrics
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_probs), np.array(all_preds)


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy, all_labels, all_probs, all_preds)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Collect all predictions for metrics calculation
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            # Forward pass
            outputs = model(images).squeeze()
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Get probabilities and predictions
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Collect for metrics
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_probs), np.array(all_preds)


def compute_classification_metrics(labels, probs, preds):
    """
    Compute classification metrics: AUC-ROC, F1, Recall, Precision.
    
    Args:
        labels: True labels (numpy array)
        probs: Predicted probabilities (numpy array)
        preds: Predicted classes (numpy array)
        
    Returns:
        dict: Dictionary with auc, f1, recall, precision
    """
    metrics = {}
    
    # AUC-ROC (requires probabilities)
    try:
        # Need at least one sample of each class for AUC
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, probs)
        else:
            metrics['auc'] = 0.0  # Cannot compute AUC with single class
    except Exception:
        metrics['auc'] = 0.0
    
    # F1-score for positive class (Case)
    try:
        metrics['f1'] = f1_score(labels, preds, pos_label=1, zero_division=0)
    except Exception:
        metrics['f1'] = 0.0
    
    # Recall for positive class (Case) - sensitivity/true positive rate
    try:
        metrics['recall'] = recall_score(labels, preds, pos_label=1, zero_division=0)
    except Exception:
        metrics['recall'] = 0.0
    
    # Precision for positive class (Case)
    try:
        metrics['precision'] = precision_score(labels, preds, pos_label=1, zero_division=0)
    except Exception:
        metrics['precision'] = 0.0
    
    return metrics


def train_single_run(train_loader, val_loader, full_loader, metadata, run_id, seed, device=None):
    """
    Train model for a single run and track cartography metrics.
    
    Stores per-epoch predictions for windowed cartography analysis.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        full_loader: Full dataset loader (for computing metrics)
        metadata: DataFrame with patient_id mapping
        run_id: Run number (1-5)
        seed: Random seed for this run
        device: Device to run on
        
    Returns:
        tuple: (epoch_metrics, epoch_predictions, training_log, best_epoch)
            - epoch_metrics: list of dicts with confidence/correctness per epoch
            - epoch_predictions: list of lists, each containing prediction dicts per epoch
            - training_log: list of dicts with loss/accuracy per epoch
            - best_epoch: 0-indexed epoch with best val_loss
    """
    if device is None:
        device = get_device()
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"RUN {run_id} (seed={seed})")
    print(f"{'='*60}")
    
    # Create fresh model for this run
    model = create_model(pretrained=True)
    model = model.to(device)
    
    # Loss function: BCEWithLogitsLoss without pos_weight
    # Using label smoothing instead of pos_weight for class imbalance
    criterion = nn.BCEWithLogitsLoss()
    print(f"Using BCEWithLogitsLoss (pos_weight=None, label_smoothing={LABEL_SMOOTHING})")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"Using AdamW optimizer (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE
    )
    print(f"Using ReduceLROnPlateau scheduler (factor={LR_SCHEDULER_FACTOR}, patience={LR_SCHEDULER_PATIENCE})")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Storage for metrics
    epoch_metrics = []       # List of dicts: {image_no: {confidence, correctness}}
    epoch_predictions = []   # List of lists: [[{image_id, patient_id, y_true, y_pred_logit}, ...], ...]
    training_log = []        # List of dicts: {epoch, train_loss, train_acc, val_loss, val_acc}
    best_model_state = None
    best_epoch = 0           # 0-indexed
    best_metrics = {}
    
    # Training loop
    print(f"\nTraining for up to {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Train one epoch (returns loss, accuracy, and predictions for metrics)
        train_loss, train_acc, train_labels, train_probs, train_preds = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate (returns loss, accuracy, and predictions for metrics)
        val_loss, val_acc, val_labels, val_probs, val_preds = validate(
            model, val_loader, criterion, device
        )
        
        # Compute additional classification metrics
        train_metrics = compute_classification_metrics(train_labels, train_probs, train_preds)
        val_metrics = compute_classification_metrics(val_labels, val_probs, val_preds)
        
        # Log training progress with all metrics
        log_entry = {
            'run': run_id,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_auc': train_metrics['auc'],
            'val_auc': val_metrics['auc'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision']
        }
        training_log.append(log_entry)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress (basic metrics + AUC + LR)
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Check early stopping and save best model
        is_best = early_stopping(val_loss)
        
        if is_best:
            # Save best model state when val_loss hits a new minimum
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch  # 0-indexed
            best_metrics = {
                'epoch': epoch + 1,  # 1-indexed for display
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision'],
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        
        # Compute confidence and correctness on FULL dataset for this epoch (legacy)
        metrics = compute_confidence_and_correctness(model, full_loader, device)
        epoch_metrics.append(metrics)
        
        # Store detailed predictions for windowed cartography analysis
        predictions = compute_epoch_predictions(model, full_loader, metadata, device)
        epoch_predictions.append(predictions)
        
        # Check if we should stop (val_loss not improving for patience epochs)
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Restoring best model checkpoint (val_loss={early_stopping.best_loss:.4f})")
            break
    
    # Restore and save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model_path = MODELS_DIR / f"model_run{run_id}.pth"
        save_model(model, model_path)
        
        # Save best model info to text file
        save_best_model_info(run_id, seed, best_metrics, len(training_log))
    
    print(f"\nRun {run_id} completed: {len(training_log)} epochs trained")
    print(f"Best epoch (0-indexed): {best_epoch} with val_loss={early_stopping.best_loss:.4f}")
    
    return epoch_metrics, epoch_predictions, training_log, best_epoch


def save_best_model_info(run_id, seed, best_metrics, total_epochs):
    """
    Save best model information to a text file.
    
    Args:
        run_id: Run number
        seed: Random seed used
        best_metrics: Dictionary with best model metrics
        total_epochs: Total epochs trained
    """
    from dataset_cartography.config import OUTPUT_DIR
    
    info_file = OUTPUT_DIR / "best_models_summary.txt"
    
    # Create header if file doesn't exist
    write_header = not info_file.exists()
    
    with open(info_file, 'a') as f:
        if write_header:
            f.write("=" * 70 + "\n")
            f.write("BEST MODEL SUMMARY FOR EACH TRAINING RUN\n")
            f.write("=" * 70 + "\n\n")
        
        f.write(f"RUN {run_id} (seed={seed})\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Best Epoch:       {best_metrics['epoch']} / {total_epochs}\n")
        f.write(f"  Validation Loss:  {best_metrics['val_loss']:.4f}\n")
        f.write(f"  Validation Acc:   {best_metrics['val_acc']:.4f}\n")
        f.write(f"  Validation AUC:   {best_metrics['val_auc']:.4f}\n")
        f.write(f"  Validation F1:    {best_metrics['val_f1']:.4f}\n")
        f.write(f"  Validation Recall:{best_metrics['val_recall']:.4f}\n")
        f.write(f"  Validation Prec:  {best_metrics['val_precision']:.4f}\n")
        f.write(f"  Training Loss:    {best_metrics['train_loss']:.4f}\n")
        f.write(f"  Training Acc:     {best_metrics['train_acc']:.4f}\n")
        f.write(f"  Model saved to:   models/model_run{run_id}.pth\n")
        f.write("\n")
    
    print(f"Best model info saved to: {info_file}")


def run_all_training(train_loader, val_loader, full_loader, metadata, seeds=None, device=None):
    """
    Run training for all seeds and collect cartography metrics.
    
    Collects both legacy metrics and new windowed cartography data.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        full_loader: Full dataset loader
        metadata: DataFrame with image metadata
        seeds: List of random seeds (default from config)
        device: Device to run on
        
    Returns:
        tuple: (confidence_matrix, correctness_matrix, all_training_logs, 
                all_epoch_predictions, best_epochs)
            - confidence_matrix: Legacy metrics {image_no: [[conf per epoch] per run]}
            - correctness_matrix: Legacy metrics {image_no: [[corr per epoch] per run]}
            - all_training_logs: Combined training logs from all runs
            - all_epoch_predictions: List of per-run epoch predictions
            - best_epochs: List of best epochs (0-indexed) per run
    """
    from dataset_cartography.config import RANDOM_SEEDS, NUM_RUNS
    
    if seeds is None:
        seeds = RANDOM_SEEDS
    if device is None:
        device = get_device()
    
    # Get all image numbers
    image_numbers = sorted(metadata['image_no'].tolist())
    num_images = len(image_numbers)
    
    # Initialize matrices (legacy format)
    # Shape: [num_images][num_epochs][num_runs]
    max_epochs = NUM_EPOCHS
    confidence_matrix = {}
    correctness_matrix = {}
    
    for img_no in image_numbers:
        confidence_matrix[img_no] = []
        correctness_matrix[img_no] = []
    
    all_training_logs = []
    all_epoch_predictions = []  # New: store epoch predictions for each run
    best_epochs = []            # New: store best epoch for each run
    
    # Run training for each seed
    for run_idx, seed in enumerate(seeds):
        run_id = run_idx + 1
        
        # Updated to pass metadata and receive new return values
        epoch_metrics, epoch_predictions, training_log, best_epoch = train_single_run(
            train_loader, val_loader, full_loader, metadata,
            run_id, seed, device
        )
        
        all_training_logs.extend(training_log)
        all_epoch_predictions.append(epoch_predictions)  # Store for windowed analysis
        best_epochs.append(best_epoch)  # Store best epoch (0-indexed)
        
        # Store legacy metrics for this run
        for img_no in image_numbers:
            conf_per_epoch = []
            corr_per_epoch = []
            
            for epoch_data in epoch_metrics:
                if img_no in epoch_data:
                    conf_per_epoch.append(epoch_data[img_no]['confidence'])
                    corr_per_epoch.append(epoch_data[img_no]['correctness'])
                else:
                    # Should not happen, but handle gracefully
                    conf_per_epoch.append(0.5)
                    corr_per_epoch.append(0)
            
            confidence_matrix[img_no].append(conf_per_epoch)
            correctness_matrix[img_no].append(corr_per_epoch)
    
    print("\n" + "="*60)
    print("ALL TRAINING RUNS COMPLETED")
    print("="*60)
    print(f"Best epochs per run (0-indexed): {best_epochs}")
    
    return confidence_matrix, correctness_matrix, all_training_logs, all_epoch_predictions, best_epochs


def save_training_logs(training_logs, output_path=None):
    """
    Save training logs to CSV file.
    
    Args:
        training_logs: List of log dictionaries
        output_path: Path to save CSV (default from config)
    """
    from dataset_cartography.config import TRAINING_LOG_CSV
    
    if output_path is None:
        output_path = LOGS_DIR / TRAINING_LOG_CSV
    
    df = pd.DataFrame(training_logs)
    df.to_csv(output_path, index=False)
    print(f"Training logs saved to: {output_path}")


if __name__ == "__main__":
    print("Training module loaded successfully.")
    print("Use run_all_training() to train models for cartography analysis.")

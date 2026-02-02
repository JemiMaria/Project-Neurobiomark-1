"""
Training module for ALS vs ALS-FTD classification.

Implements:
- EfficientNetB0 with ImageNet pretrained weights
- BCEWithLogitsLoss (no pos_weight - balanced classes)
- AdamW optimizer
- ReduceLROnPlateau scheduler
- Early stopping on validation loss
- Per-epoch prediction logging for cartography
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from config import (
    DEVICE, IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, MIN_DELTA,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
    DROPOUT_RATE, IMAGENET_MEAN, IMAGENET_STD,
    LOGS_DIR, CHECKPOINTS_DIR, RANDOM_SEEDS
)
from utils import set_seed


class ALSDataset(Dataset):
    """
    Dataset for ALS vs ALS-FTD classification.
    
    Args:
        metadata_df: DataFrame with [image_path, patient_id, y_true]
        transform: torchvision transforms
        image_dir: Base directory for images
    """
    
    def __init__(self, metadata_df, transform=None, image_dir=None):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = Path(image_dir) if image_dir else IMAGE_DIR
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['y_true'], dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'patient_id': row['patient_id'],
            'image_path': row['image_path'],
            'idx': idx
        }


def get_train_transforms():
    """Get training transforms with light augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_model(dropout_rate=None):
    """
    Create EfficientNetB0 model with custom classifier.
    
    Args:
        dropout_rate: Dropout rate before classifier
        
    Returns:
        nn.Module: Model
    """
    if dropout_rate is None:
        dropout_rate = DROPOUT_RATE
    
    # Load pretrained EfficientNetB0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, 1)  # Binary classification
    )
    
    return model


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            return True
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_fold(train_df, test_df, fold_id, seed, config=None, verbose=True):
    """
    Train model for one LOPO fold with one seed.
    
    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame (held-out patient)
        fold_id: Fold ID
        seed: Random seed
        config: Optional config override
        verbose: Print progress
        
    Returns:
        dict: {
            'model': trained model,
            'history': training history,
            'epoch_predictions': per-epoch predictions for cartography,
            'best_epoch': best validation epoch
        }
    """
    set_seed(seed)
    
    if verbose:
        print(f"\n  Training Fold {fold_id}, Seed {seed}")
        print(f"    Train: {len(train_df)} images from {train_df['patient_id'].nunique()} patients")
        print(f"    Test: {len(test_df)} images from {test_df['patient_id'].nunique()} patient")
    
    # Create datasets
    train_dataset = ALSDataset(train_df, transform=get_train_transforms())
    test_dataset = ALSDataset(test_df, transform=get_val_transforms())
    
    # Create dataloaders (no weighted sampling - balanced classes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Create model
    model = create_model()
    model = model.to(DEVICE)
    
    # Loss function (no pos_weight - balanced classes)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Store per-epoch predictions for cartography
    epoch_predictions = []
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        train_loss /= n_train_batches
        
        # Validation phase (on held-out patient)
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        # Collect predictions for this epoch
        epoch_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                n_val_batches += 1
                
                # Store predictions for cartography
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                logits = outputs.cpu().numpy().flatten()
                
                for i, (img_path, patient_id, y_true) in enumerate(zip(
                    batch['image_path'], batch['patient_id'], batch['label'].numpy()
                )):
                    epoch_preds.append({
                        'epoch': epoch,
                        'image_path': img_path,
                        'patient_id': patient_id,
                        'y_true': int(y_true),
                        'logit': float(logits[i]),
                        'prob': float(probs[i])
                    })
        
        val_loss /= n_val_batches if n_val_batches > 0 else 1
        epoch_predictions.extend(epoch_preds)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # Check early stopping
        improved = early_stopping(val_loss, model)
        
        # Print per-epoch metrics
        if verbose:
            status = "✓ (best)" if improved else " "
            print(f"      Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} {status}")
        
        if early_stopping.early_stop:
            if verbose:
                print(f"      Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    early_stopping.restore_best_model(model)
    
    # Find best epoch
    best_epoch = np.argmin(history['val_loss'])
    
    if verbose:
        print(f"      Best epoch: {best_epoch+1}, Val loss: {history['val_loss'][best_epoch]:.4f}")
    
    # Save training log
    log_df = pd.DataFrame(history)
    log_path = LOGS_DIR / f"fold_{fold_id}_seed_{seed}_training_log.csv"
    log_df.to_csv(log_path, index=False)
    
    # Save checkpoint
    checkpoint_path = CHECKPOINTS_DIR / f"best_model_fold_{fold_id}_seed_{seed}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'fold_id': fold_id,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': history['val_loss'][best_epoch]
    }, checkpoint_path)
    
    return {
        'model': model,
        'history': history,
        'epoch_predictions': epoch_predictions,
        'best_epoch': best_epoch
    }


def get_predictions_for_patient(model, test_loader, device=None):
    """
    Get predictions for held-out patient.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test patient
        device: torch device
        
    Returns:
        dict: {'patient_id', 'y_true', 'mean_prob', 'predictions': [...]}
    """
    if device is None:
        device = DEVICE
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            for i in range(len(probs)):
                predictions.append({
                    'image_path': batch['image_path'][i],
                    'patient_id': batch['patient_id'][i],
                    'y_true': int(batch['label'][i].item()),
                    'prob': float(probs[i])
                })
    
    if not predictions:
        return None
    
    # Aggregate to patient level
    patient_id = predictions[0]['patient_id']
    y_true = predictions[0]['y_true']
    mean_prob = np.mean([p['prob'] for p in predictions])
    
    return {
        'patient_id': patient_id,
        'y_true': y_true,
        'mean_prob': mean_prob,
        'predictions': predictions
    }


def train_all_folds(df):
    """
    Train model across all LOPO folds and seeds.
    
    Args:
        df: DataFrame with all data (image_path, patient_id, label)
        
    Returns:
        dict: {(fold, seed): epoch_predictions} for cartography
    """
    from lopo_splits import create_lopo_splits
    
    print(f"\n{'='*60}")
    print("TRAINING ALL FOLDS")
    print(f"{'='*60}")
    
    # Get unique patients
    patients = df['patient_id'].unique()
    n_folds = len(patients)
    
    print(f"\n  Total patients: {n_folds}")
    print(f"  Seeds: {RANDOM_SEEDS}")
    print(f"  Total training runs: {n_folds * len(RANDOM_SEEDS)}")
    
    all_epoch_predictions = {}
    
    for fold_idx, test_patient in enumerate(patients):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}: Hold out patient {test_patient}")
        print(f"{'='*60}")
        
        # Split data
        test_df = df[df['patient_id'] == test_patient].copy()
        train_df = df[df['patient_id'] != test_patient].copy()
        
        print(f"  Train: {len(train_df)} images from {train_df['patient_id'].nunique()} patients")
        print(f"  Test:  {len(test_df)} images from patient {test_patient}")
        
        for seed in RANDOM_SEEDS:
            print(f"\n  --- Seed {seed} ---")
            
            # Train this fold-seed combination
            results = train_fold(
                train_df=train_df,
                test_df=test_df,
                fold_id=fold_idx,
                seed=seed,
                verbose=True
            )
            
            # Store epoch predictions and history for cartography
            all_epoch_predictions[(fold_idx, seed)] = {
                'epoch_predictions': results['epoch_predictions'],
                'history': results['history']
            }
            
            # Print training summary
            print(f"    Best epoch: {results['best_epoch']}")
            best_loss = results['history']['val_loss'][results['best_epoch']]
            print(f"    Best val_loss: {best_loss:.4f}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Completed {len(all_epoch_predictions)} fold-seed combinations")
    
    return all_epoch_predictions

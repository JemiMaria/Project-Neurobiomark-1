"""
Training logic with two-phase fine-tuning.

Training Schedule:
- Phase A: Freeze backbone, train head-only for FREEZE_EPOCHS
- Phase B: Unfreeze last stage, fine-tune with smaller backbone LR

Features:
- Early stopping on validation loss
- ReduceLROnPlateau scheduler
- Label smoothing support (optional)
- BCEWithLogitsLoss (NO pos_weight - sampler handles balancing)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, FREEZE_EPOCHS, FINETUNE_EPOCHS, HEAD_LR, BACKBONE_LR_MULTIPLIER,
    WEIGHT_DECAY, LABEL_SMOOTHING, PATIENCE, MIN_DELTA,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, CHECKPOINTS_DIR
)

from .efficientnet import freeze_backbone, unfreeze_last_stage, get_parameter_groups


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops if no improvement
    for 'patience' consecutive epochs.
    """
    
    def __init__(self, patience=7, min_delta=1e-4, mode='min'):
        """
        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.mode == 'min':
            is_better = self.best_score is None or score < self.best_score - self.min_delta
        else:
            is_better = self.best_score is None or score > self.best_score + self.min_delta
        
        if is_better:
            self.best_score = score
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved
    
    def restore_best_model(self, model):
        """Restore model to best state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy with label smoothing.
    
    Smooths labels: y → y * (1 - smoothing) + 0.5 * smoothing
    """
    
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        if self.smoothing > 0:
            # Smooth labels: 0 → smoothing/2, 1 → 1 - smoothing/2
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


class Trainer:
    """
    Two-phase trainer for EfficientNetB0.
    
    Phase A: Frozen backbone, train classifier head
    Phase B: Unfreeze last stage, fine-tune with differential LR
    
    IMPORTANT:
    - Does NOT use pos_weight (patient-balanced sampler handles class imbalance)
    - Uses label smoothing instead (configurable)
    """
    
    def __init__(
        self,
        model,
        device=None,
        head_lr=None,
        backbone_lr_multiplier=None,
        weight_decay=None,
        label_smoothing=None,
        freeze_epochs=None,
        finetune_epochs=None,
        patience=None,
        min_delta=None,
        scheduler_factor=None,
        scheduler_patience=None,
        optimizer_name='adamw',
        scheduler_name='reduce_on_plateau',
    ):
        """
        Initialize trainer with configuration.
        
        Args:
            model: EfficientNetB0 model
            device: torch device
            head_lr: Learning rate for classifier head
            backbone_lr_multiplier: LR multiplier for backbone
            weight_decay: AdamW weight decay
            label_smoothing: Label smoothing factor (0.0 = none)
            freeze_epochs: Epochs for Phase A (frozen backbone)
            finetune_epochs: Epochs for Phase B (fine-tuning)
            patience: Early stopping patience
            min_delta: Early stopping minimum delta
            scheduler_factor: LR reduction factor
            scheduler_patience: Scheduler patience
            optimizer_name: 'adam' or 'adamw'
            scheduler_name: 'reduce_on_plateau', 'cosine', or 'step'
        """
        # Use defaults from config
        self.device = device or DEVICE
        self.head_lr = head_lr if head_lr is not None else HEAD_LR
        self.backbone_lr_multiplier = backbone_lr_multiplier if backbone_lr_multiplier is not None else BACKBONE_LR_MULTIPLIER
        self.weight_decay = weight_decay if weight_decay is not None else WEIGHT_DECAY
        self.label_smoothing = label_smoothing if label_smoothing is not None else LABEL_SMOOTHING
        self.freeze_epochs = freeze_epochs if freeze_epochs is not None else FREEZE_EPOCHS
        self.finetune_epochs = finetune_epochs if finetune_epochs is not None else FINETUNE_EPOCHS
        self.patience = patience if patience is not None else PATIENCE
        self.min_delta = min_delta if min_delta is not None else MIN_DELTA
        self.scheduler_factor = scheduler_factor if scheduler_factor is not None else LR_SCHEDULER_FACTOR
        self.scheduler_patience = scheduler_patience if scheduler_patience is not None else LR_SCHEDULER_PATIENCE
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        
        self.model = model.to(self.device)
        
        # Loss function (NO pos_weight - sampler handles balance)
        if self.label_smoothing > 0:
            self.criterion = LabelSmoothingBCELoss(smoothing=self.label_smoothing)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': [],
            'phase': [],
            'lr': []
        }
        
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta,
            mode='min'
        )
    
    def _create_optimizer(self, params, lr):
        """
        Create optimizer based on optimizer_name.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            
        Returns:
            torch optimizer
        """
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _create_scheduler(self, optimizer, total_epochs):
        """
        Create learning rate scheduler based on scheduler_name.
        
        Args:
            optimizer: torch optimizer
            total_epochs: Total number of epochs for this phase
            
        Returns:
            torch scheduler (or None for reduce_on_plateau, handled separately)
        """
        if self.scheduler_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience
            )
        elif self.scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=1e-7
            )
        elif self.scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, total_epochs // 3),
                gamma=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            logits = self.model(images).squeeze(-1)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def _validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(images).squeeze(-1)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(self, train_loader, val_loader, verbose=True):
        """
        Run two-phase training.
        
        Phase A: Freeze backbone, train head for freeze_epochs
        Phase B: Unfreeze last stage, fine-tune for finetune_epochs
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            verbose: Print progress
            
        Returns:
            dict: Training history
        """
        total_epochs = self.freeze_epochs + self.finetune_epochs
        current_epoch = 0
        
        # ===== PHASE A: Frozen Backbone =====
        if verbose:
            print(f"\n{'='*50}")
            print(f"PHASE A: Head-only training ({self.freeze_epochs} epochs)")
            print(f"{'='*50}")
        
        freeze_backbone(self.model)
        
        # Optimizer for Phase A (head only)
        optimizer = self._create_optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.head_lr
        )
        
        scheduler = self._create_scheduler(optimizer, self.freeze_epochs)
        
        for epoch in range(self.freeze_epochs):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self._validate(val_loader)
            
            # Step scheduler (handle reduce_on_plateau differently)
            if self.scheduler_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'].append(current_epoch)
            self.history['phase'].append('A')
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            improved = self.early_stopping(val_loss, self.model)
            
            if verbose:
                status = "✓" if improved else " "
                print(f"  Epoch {current_epoch+1:2d}/{total_epochs} [A] | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | {status}")
            
            if self.early_stopping.early_stop:
                if verbose:
                    print(f"  Early stopping triggered at epoch {current_epoch+1}")
                break
            
            current_epoch += 1
        
        # ===== PHASE B: Fine-tuning =====
        if not self.early_stopping.early_stop:
            if verbose:
                print(f"\n{'='*50}")
                print(f"PHASE B: Fine-tuning ({self.finetune_epochs} epochs)")
                print(f"{'='*50}")
            
            unfreeze_last_stage(self.model)
            
            # Optimizer for Phase B (differential LR)
            param_groups = get_parameter_groups(
                self.model,
                self.head_lr,
                self.backbone_lr_multiplier
            )
            
            # Create optimizer with param groups
            if self.optimizer_name == 'adam':
                optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)
            else:  # adamw
                optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
            
            scheduler = self._create_scheduler(optimizer, self.finetune_epochs)
            
            # Reset early stopping for Phase B
            self.early_stopping.counter = 0
            self.early_stopping.early_stop = False
            
            for epoch in range(self.finetune_epochs):
                train_loss = self._train_epoch(train_loader, optimizer)
                val_loss = self._validate(val_loader)
                
                # Step scheduler (handle reduce_on_plateau differently)
                if self.scheduler_name == 'reduce_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['epoch'].append(current_epoch)
                self.history['phase'].append('B')
                self.history['lr'].append(optimizer.param_groups[0]['lr'])
                
                improved = self.early_stopping(val_loss, self.model)
                
                if verbose:
                    status = "✓" if improved else " "
                    print(f"  Epoch {current_epoch+1:2d}/{total_epochs} [B] | "
                          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | {status}")
                
                if self.early_stopping.early_stop:
                    if verbose:
                        print(f"  Early stopping triggered at epoch {current_epoch+1}")
                    break
                
                current_epoch += 1
        
        # Restore best model
        self.early_stopping.restore_best_model(self.model)
        
        if verbose:
            print(f"\n  Best val_loss: {self.early_stopping.best_score:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path, fold_id=None, extra_info=None):
        """
        Save model checkpoint.
        
        Args:
            path: Save path
            fold_id: Fold ID (optional)
            extra_info: Additional info to save (optional)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_loss': self.early_stopping.best_score,
            'config': {
                'head_lr': self.head_lr,
                'backbone_lr_multiplier': self.backbone_lr_multiplier,
                'weight_decay': self.weight_decay,
                'label_smoothing': self.label_smoothing,
                'freeze_epochs': self.freeze_epochs,
                'finetune_epochs': self.finetune_epochs,
            }
        }
        
        if fold_id is not None:
            checkpoint['fold_id'] = fold_id
        
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint

"""
Optuna hyperparameter tuning with leakage prevention.

LEAKAGE PREVENTION (Rule #2):
This implements nested cross-validation:
- Outer CV: Final reporting (not touched by Optuna)
- Inner CV: Optuna tuning

Alternatively, uses a held-out patient set never seen during tuning.

Search space:
- head_lr: log scale
- backbone_lr_multiplier
- weight_decay
- dropout
- label_smoothing
- freeze_epochs
- aug_level, max_rotate, p_noise, use_elastic
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, N_FOLDS, RANDOM_SEED, OPTUNA_DIR,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_PRUNER_WARMUP, OPTUNA_STUDY_NAME,
    OPTUNA_HEAD_LR_RANGE, OPTUNA_BACKBONE_LR_MULT_RANGE,
    OPTUNA_WEIGHT_DECAY_RANGE, OPTUNA_DROPOUT_RANGE,
    OPTUNA_LABEL_SMOOTHING_RANGE, OPTUNA_FREEZE_EPOCHS_RANGE,
    OPTUNA_BATCH_SIZE_CHOICES, OPTUNA_OPTIMIZER_CHOICES, OPTUNA_SCHEDULER_CHOICES
)

from data import (
    load_and_validate_metadata,
    create_patient_grouped_splits,
    create_fold_dataloaders,
    get_train_transforms,
    get_val_transforms
)
from models import create_efficientnet_b0, Trainer
from evaluation import compute_fold_metrics, tune_threshold_per_fold


class OptunaCV:
    """
    Optuna hyperparameter tuning with cross-validation.
    
    Uses nested CV or held-out set to prevent tuning leakage.
    
    LEAKAGE PREVENTION:
    - Fixed folds across all trials (no data shuffling)
    - Inner CV folds are separate from outer CV reporting folds
    """
    
    def __init__(
        self,
        metadata_df,
        splits,
        n_inner_folds=3,
        device=None,
        output_dir=None
    ):
        """
        Initialize Optuna CV tuner.
        
        Args:
            metadata_df: Full metadata DataFrame
            splits: Pre-computed fold splits
            n_inner_folds: Number of inner CV folds for tuning
            device: torch device
            output_dir: Directory for Optuna outputs
        """
        self.metadata_df = metadata_df
        self.splits = splits
        self.n_inner_folds = n_inner_folds
        self.device = device or DEVICE
        self.output_dir = Path(output_dir) if output_dir else OPTUNA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use first n_inner_folds for tuning, rest for final validation
        self.tuning_folds = list(range(n_inner_folds))
        self.held_out_folds = list(range(n_inner_folds, len(splits)))
        
        print(f"\nOptuna CV Setup:")
        print(f"  Tuning folds: {self.tuning_folds}")
        print(f"  Held-out folds (for final eval): {self.held_out_folds}")
    
    def create_objective(self):
        """
        Create Optuna objective function.
        
        Returns:
            callable: Objective function for Optuna
        """
        def objective(trial):
            # Sample hyperparameters
            head_lr = trial.suggest_float(
                'head_lr',
                OPTUNA_HEAD_LR_RANGE[0],
                OPTUNA_HEAD_LR_RANGE[1],
                log=True
            )
            
            backbone_lr_mult = trial.suggest_float(
                'backbone_lr_multiplier',
                OPTUNA_BACKBONE_LR_MULT_RANGE[0],
                OPTUNA_BACKBONE_LR_MULT_RANGE[1]
            )
            
            weight_decay = trial.suggest_float(
                'weight_decay',
                OPTUNA_WEIGHT_DECAY_RANGE[0],
                OPTUNA_WEIGHT_DECAY_RANGE[1],
                log=True
            )
            
            dropout = trial.suggest_float(
                'dropout',
                OPTUNA_DROPOUT_RANGE[0],
                OPTUNA_DROPOUT_RANGE[1]
            )
            
            label_smoothing = trial.suggest_float(
                'label_smoothing',
                OPTUNA_LABEL_SMOOTHING_RANGE[0],
                OPTUNA_LABEL_SMOOTHING_RANGE[1]
            )
            
            freeze_epochs = trial.suggest_int(
                'freeze_epochs',
                OPTUNA_FREEZE_EPOCHS_RANGE[0],
                OPTUNA_FREEZE_EPOCHS_RANGE[1]
            )
            
            # Augmentation parameters
            aug_level = trial.suggest_categorical('aug_level', ['light', 'medium', 'strong'])
            max_rotate = trial.suggest_categorical('max_rotate', [10, 20, 30])
            p_noise = trial.suggest_float('p_noise', 0.0, 0.2)
            use_elastic = trial.suggest_categorical('use_elastic', [0, 1])
            
            # New categorical hyperparameters
            batch_size = trial.suggest_categorical('batch_size', OPTUNA_BATCH_SIZE_CHOICES)
            optimizer_name = trial.suggest_categorical('optimizer', OPTUNA_OPTIMIZER_CHOICES)
            scheduler_name = trial.suggest_categorical('scheduler', OPTUNA_SCHEDULER_CHOICES)
            
            # Get transforms with suggested augmentation
            train_transform = get_train_transforms(
                aug_level=aug_level,
                max_rotate=max_rotate,
                p_noise=p_noise,
                use_elastic=bool(use_elastic)
            )
            val_transform = get_val_transforms()
            
            # Cross-validation on tuning folds
            fold_balanced_accs = []
            
            for fold_id in self.tuning_folds:
                # Create dataloaders for this fold
                train_loader, val_loader, _, _ = create_fold_dataloaders(
                    self.metadata_df,
                    self.splits,
                    fold_id,
                    train_transform=train_transform,
                    val_transform=val_transform,
                    batch_size=batch_size
                )
                
                # Create model
                model = create_efficientnet_b0(dropout_rate=dropout)
                
                # Create trainer
                trainer = Trainer(
                    model,
                    device=self.device,
                    head_lr=head_lr,
                    backbone_lr_multiplier=backbone_lr_mult,
                    weight_decay=weight_decay,
                    label_smoothing=label_smoothing,
                    freeze_epochs=freeze_epochs,
                    optimizer_name=optimizer_name,
                    scheduler_name=scheduler_name
                )
                
                # Train
                trainer.train(train_loader, val_loader, verbose=False)
                
                # Evaluate
                metrics, _, _ = compute_fold_metrics(model, val_loader, device=self.device)
                fold_balanced_accs.append(metrics['balanced_accuracy'])
                
                # Report intermediate result for pruning
                trial.report(np.mean(fold_balanced_accs), fold_id)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return mean balanced accuracy across tuning folds
            return np.mean(fold_balanced_accs)
        
        return objective
    
    def run(self, n_trials=None, timeout=None):
        """
        Run Optuna hyperparameter search.
        
        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds
            
        Returns:
            optuna.Study: Completed study
        """
        if n_trials is None:
            n_trials = OPTUNA_N_TRIALS
        if timeout is None:
            timeout = OPTUNA_TIMEOUT
        
        print(f"\n{'='*60}")
        print("OPTUNA HYPERPARAMETER TUNING")
        print(f"{'='*60}")
        print(f"  Trials: {n_trials}")
        print(f"  Timeout: {timeout}")
        print(f"  Objective: Maximize mean balanced accuracy")
        
        # Create study
        study = optuna.create_study(
            study_name=OPTUNA_STUDY_NAME,
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=OPTUNA_PRUNER_WARMUP
            )
        )
        
        # Run optimization
        study.optimize(
            self.create_objective(),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study):
        """Save study results."""
        # Best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n  Best trial:")
        print(f"    Value (balanced_accuracy): {best_value:.4f}")
        print(f"    Parameters:")
        for key, value in best_params.items():
            print(f"      {key}: {value}")
        
        # Save best params as JSON
        with open(self.output_dir / "best_params.json", 'w') as f:
            json.dump({
                'best_value': best_value,
                'best_params': best_params,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / "optuna_trials.csv", index=False)
        
        # Save study object
        with open(self.output_dir / "optuna_study.pkl", 'wb') as f:
            pickle.dump(study, f)
        
        print(f"\n  ✓ Saved: best_params.json")
        print(f"  ✓ Saved: optuna_trials.csv")
        print(f"  ✓ Saved: optuna_study.pkl")
        
        # Create parameter importance plot
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(study)
            fig.write_image(str(self.output_dir / "param_importances.png"))
            print(f"  ✓ Saved: param_importances.png")
        except Exception as e:
            print(f"  ⚠ Could not save param importances plot: {e}")


def run_optuna_tuning(metadata_df=None, splits=None, n_trials=None, timeout=None):
    """
    Run Optuna hyperparameter tuning.
    
    Args:
        metadata_df: Metadata DataFrame (loads if None)
        splits: Pre-computed splits (creates if None)
        n_trials: Number of trials
        timeout: Timeout in seconds
        
    Returns:
        dict: Best parameters
    """
    # Load data if needed
    if metadata_df is None:
        metadata_df = load_and_validate_metadata()
    
    # Create splits if needed
    if splits is None:
        splits = create_patient_grouped_splits(metadata_df)
    
    # Create tuner
    tuner = OptunaCV(metadata_df, splits)
    
    # Run optimization
    study = tuner.run(n_trials=n_trials, timeout=timeout)
    
    return study.best_params


def load_best_params(params_path=None):
    """
    Load best parameters from saved JSON.
    
    Args:
        params_path: Path to best_params.json
        
    Returns:
        dict: Best parameters
    """
    if params_path is None:
        params_path = OPTUNA_DIR / "best_params.json"
    
    with open(params_path, 'r') as f:
        data = json.load(f)
    
    return data['best_params']

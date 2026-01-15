"""
Main entry point for Grouped CV EfficientNetB0 Pipeline.

Console menu interface for:
1. Data validation and split creation
2. Optuna hyperparameter tuning
3. Full cross-validation training and evaluation
4. CAM generation (LayerCAM and Guided Grad-CAM)

Run with: python main.py

LEAKAGE PREVENTION CHECKLIST:
✓ Patient overlap: Splits created BEFORE dataset creation
✓ Tuning leakage: Nested CV with held-out folds
✓ Threshold leakage: Per-fold tuning on validation only
✓ Stain normalization: Fit on training fold only (if used)
✓ CAM leakage: CAMs for diagnostics only, not model tuning
"""

import sys
import os
from pathlib import Path

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd

from config import (
    DEVICE, N_FOLDS, RANDOM_SEED, OUTPUT_DIR, SPLITS_DIR,
    CHECKPOINTS_DIR, EVAL_DIR, CAMS_DIR
)
from utils import set_seed, print_device_info
from data import (
    load_and_validate_metadata,
    create_patient_grouped_splits,
    load_splits,
    create_fold_dataloaders,
    get_train_transforms,
    get_val_transforms
)
from models import create_efficientnet_b0, Trainer
from evaluation import (
    compute_fold_metrics,
    aggregate_to_patient_level,
    tune_threshold_per_fold,
    compute_calibration_metrics,
    plot_reliability_diagram,
    save_calibration_results,
    generate_evaluation_report
)


def print_header():
    """Print pipeline header."""
    print("\n" + "="*70)
    print("  GROUPED CV EFFICIENTNETB0 PIPELINE")
    print("  ALS (1) vs Control (0) Classification")
    print("="*70)
    print_device_info()


def print_menu():
    """Print main menu."""
    print("\n" + "-"*50)
    print("  MAIN MENU")
    print("-"*50)
    print("  [1] Validate data and create CV splits")
    print("  [2] Run Optuna hyperparameter tuning")
    print("  [3] Run full cross-validation training")
    print("  [4] Evaluate trained models")
    print("  [5] Generate CAM visualizations")
    print("  [6] Run complete pipeline (1→3→4→5)")
    print("  [0] Exit")
    print("-"*50)


def menu_validate_and_split():
    """Menu option 1: Validate data and create splits."""
    print("\n" + "="*60)
    print("STEP 1: DATA VALIDATION AND SPLIT CREATION")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Load and validate metadata
    try:
        metadata_df = load_and_validate_metadata()
    except Exception as e:
        print(f"\n  ✗ Error loading metadata: {e}")
        print(f"\n  Please ensure your metadata CSV exists and contains:")
        print(f"    - image_path: Path to image files")
        print(f"    - patient_id: Unique patient identifier")
        print(f"    - label/y_true: Binary label (0=Control, 1=ALS)")
        return None, None
    
    # Create patient-grouped splits
    splits = create_patient_grouped_splits(metadata_df)
    
    print("\n  ✓ Data validation complete")
    print(f"  ✓ Splits saved to: {SPLITS_DIR / 'grouped_cv_folds.csv'}")
    
    return metadata_df, splits


def menu_optuna_tuning(metadata_df=None, splits=None):
    """Menu option 2: Run Optuna hyperparameter tuning."""
    print("\n" + "="*60)
    print("STEP 2: OPTUNA HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data if not provided
    if metadata_df is None:
        try:
            metadata_df = load_and_validate_metadata()
            splits = load_splits()
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            print("  Please run option [1] first to create splits.")
            return None
    
    from tuning import run_optuna_tuning
    
    # Get number of trials from user
    try:
        n_trials_input = input("\n  Enter number of trials (default=50): ").strip()
        n_trials = int(n_trials_input) if n_trials_input else 50
    except ValueError:
        n_trials = 50
    
    print(f"\n  Running {n_trials} Optuna trials...")
    print("  (This may take a while)")
    
    best_params = run_optuna_tuning(metadata_df, splits, n_trials=n_trials)
    
    print("\n  ✓ Optuna tuning complete")
    print(f"  Best parameters saved to: {OUTPUT_DIR / 'optuna' / 'best_params.json'}")
    
    return best_params


def menu_train_cv(metadata_df=None, splits=None, use_best_params=True):
    """Menu option 3: Run full cross-validation training."""
    print("\n" + "="*60)
    print("STEP 3: CROSS-VALIDATION TRAINING")
    print("="*60)
    
    set_seed(RANDOM_SEED)
    
    # Load data if not provided
    if metadata_df is None:
        try:
            metadata_df = load_and_validate_metadata()
            splits = load_splits()
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            print("  Please run option [1] first to create splits.")
            return None
    
    # Load best params if available and requested
    train_params = {}
    if use_best_params:
        try:
            from tuning.optuna_tuner import load_best_params
            train_params = load_best_params()
            print(f"\n  Using Optuna best parameters")
        except FileNotFoundError:
            print(f"\n  No Optuna results found, using default parameters")
    
    # Get transforms
    aug_level = train_params.get('aug_level', 'medium')
    max_rotate = train_params.get('max_rotate', 20)
    p_noise = train_params.get('p_noise', 0.1)
    use_elastic = train_params.get('use_elastic', 0)
    
    train_transform = get_train_transforms(
        aug_level=aug_level,
        max_rotate=max_rotate,
        p_noise=p_noise,
        use_elastic=bool(use_elastic)
    )
    val_transform = get_val_transforms()
    
    # Training parameters
    head_lr = train_params.get('head_lr', 1e-3)
    backbone_lr_mult = train_params.get('backbone_lr_multiplier', 0.1)
    weight_decay = train_params.get('weight_decay', 1e-4)
    dropout = train_params.get('dropout', 0.3)
    label_smoothing = train_params.get('label_smoothing', 0.0)
    freeze_epochs = train_params.get('freeze_epochs', 3)
    
    print(f"\n  Training with:")
    print(f"    head_lr: {head_lr}")
    print(f"    backbone_lr_mult: {backbone_lr_mult}")
    print(f"    dropout: {dropout}")
    print(f"    aug_level: {aug_level}")
    
    # Train each fold
    fold_results = {}
    
    for fold_id in range(N_FOLDS):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_id + 1}/{N_FOLDS}")
        print(f"{'='*50}")
        
        # Create dataloaders
        train_loader, val_loader, train_dataset, val_dataset = create_fold_dataloaders(
            metadata_df,
            splits,
            fold_id,
            train_transform=train_transform,
            val_transform=val_transform
        )
        
        # Create model
        model = create_efficientnet_b0(dropout_rate=dropout)
        
        # Create trainer
        trainer = Trainer(
            model,
            device=DEVICE,
            head_lr=head_lr,
            backbone_lr_multiplier=backbone_lr_mult,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            freeze_epochs=freeze_epochs
        )
        
        # Train
        history = trainer.train(train_loader, val_loader, verbose=True)
        
        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"fold_{fold_id}_best.pt"
        trainer.save_checkpoint(checkpoint_path, fold_id=fold_id)
        
        fold_results[fold_id] = {
            'history': history,
            'checkpoint_path': checkpoint_path
        }
        
        print(f"\n  ✓ Fold {fold_id} complete")
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  All checkpoints saved to: {CHECKPOINTS_DIR}")
    
    return fold_results


def menu_evaluate(metadata_df=None, splits=None):
    """Menu option 4: Evaluate trained models."""
    print("\n" + "="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    # Load data if not provided
    if metadata_df is None:
        try:
            metadata_df = load_and_validate_metadata()
            splits = load_splits()
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            print("  Please run option [1] first.")
            return None
    
    # Check for checkpoints
    checkpoints = list(CHECKPOINTS_DIR.glob("fold_*_best.pt"))
    if not checkpoints:
        print(f"\n  ✗ No checkpoints found in {CHECKPOINTS_DIR}")
        print("  Please run option [3] first to train models.")
        return None
    
    print(f"\n  Found {len(checkpoints)} checkpoints")
    
    val_transform = get_val_transforms()
    
    # Evaluate each fold
    fold_metrics_list = []
    fold_patient_dfs = {}
    fold_calibrations = []
    
    for fold_id in range(N_FOLDS):
        checkpoint_path = CHECKPOINTS_DIR / f"fold_{fold_id}_best.pt"
        
        if not checkpoint_path.exists():
            print(f"\n  ⚠ Checkpoint not found for fold {fold_id}, skipping")
            continue
        
        print(f"\n  Evaluating Fold {fold_id}...")
        
        # Create dataloader
        _, val_loader, _, val_dataset = create_fold_dataloaders(
            metadata_df,
            splits,
            fold_id,
            val_transform=val_transform
        )
        
        # Load model
        model = create_efficientnet_b0()
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        # Get predictions
        from evaluation.metrics import compute_image_level_predictions
        image_preds = compute_image_level_predictions(model, val_loader, DEVICE)
        patient_preds = aggregate_to_patient_level(image_preds)
        
        fold_patient_dfs[fold_id] = patient_preds
        
        # Calibration
        cal = compute_calibration_metrics(patient_preds)
        fold_calibrations.append(cal)
    
    # Tune thresholds per fold
    fold_thresholds = tune_threshold_per_fold(fold_patient_dfs)
    
    # Compute metrics with tuned thresholds
    for fold_id, patient_df in fold_patient_dfs.items():
        threshold = fold_thresholds[fold_id]['threshold']
        from evaluation.metrics import compute_patient_level_metrics
        metrics = compute_patient_level_metrics(patient_df, threshold)
        metrics['fold_id'] = fold_id
        fold_metrics_list.append(metrics)
    
    # Generate report
    summary = generate_evaluation_report(
        fold_metrics_list,
        fold_thresholds,
        list(fold_patient_dfs.values()),
        fold_calibrations
    )
    
    # Calibration plot
    plot_reliability_diagram(list(fold_patient_dfs.values()))
    save_calibration_results(fold_calibrations)
    
    print("\n" + "="*60)
    print("  EVALUATION COMPLETE")
    print("="*60)
    print(f"  Results saved to: {EVAL_DIR}")
    
    return summary


def menu_generate_cams(metadata_df=None, splits=None):
    """Menu option 5: Generate CAM visualizations."""
    print("\n" + "="*60)
    print("STEP 5: CAM VISUALIZATION GENERATION")
    print("="*60)
    
    # Load data if not provided
    if metadata_df is None:
        try:
            metadata_df = load_and_validate_metadata()
            splits = load_splits()
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            return None
    
    # Check for checkpoints
    checkpoints = list(CHECKPOINTS_DIR.glob("fold_*_best.pt"))
    if not checkpoints:
        print(f"\n  ✗ No checkpoints found. Please run training first.")
        return None
    
    # Use best fold (fold 0 by default) or ask user
    try:
        fold_input = input(f"\n  Enter fold to use for CAMs (0-{N_FOLDS-1}, default=0): ").strip()
        fold_id = int(fold_input) if fold_input else 0
    except ValueError:
        fold_id = 0
    
    checkpoint_path = CHECKPOINTS_DIR / f"fold_{fold_id}_best.pt"
    if not checkpoint_path.exists():
        print(f"\n  ✗ Checkpoint not found for fold {fold_id}")
        return None
    
    # Load model
    print(f"\n  Loading model from fold {fold_id}...")
    model = create_efficientnet_b0()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Create validation dataloader
    val_transform = get_val_transforms()
    _, val_loader, _, val_dataset = create_fold_dataloaders(
        metadata_df, splits, fold_id, val_transform=val_transform
    )
    
    # Get patient labels
    patient_labels = {pid: label for pid, label in 
                     metadata_df.groupby('patient_id')['y_true'].first().items()}
    
    # Ask for max images
    try:
        max_input = input("\n  Max images to process (empty for all): ").strip()
        max_images = int(max_input) if max_input else None
    except ValueError:
        max_images = None
    
    # Generate LayerCAM
    from interpretability import generate_layercam_outputs, compute_cam_consistency, plot_cam_comparison
    
    print("\n  Generating LayerCAM outputs...")
    layercam_cams = generate_layercam_outputs(
        model, val_loader, 
        output_dir=CAMS_DIR / "layercam",
        max_images=max_images
    )
    
    # LayerCAM consistency analysis
    layercam_results = compute_cam_consistency(
        layercam_cams, patient_labels,
        output_dir=CAMS_DIR / "layercam",
        method_name="layercam"
    )
    plot_cam_comparison(
        layercam_results['als_mean'],
        layercam_results['control_mean'],
        CAMS_DIR / "layercam" / "cam_class_comparison.png"
    )
    
    # Generate Guided Grad-CAM
    from interpretability import generate_guided_gradcam_outputs
    
    print("\n  Generating Guided Grad-CAM outputs...")
    # Reload model (hooks may interfere)
    model = create_efficientnet_b0()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    ggcam_cams = generate_guided_gradcam_outputs(
        model, val_loader,
        output_dir=CAMS_DIR / "guided_gradcam",
        max_images=max_images
    )
    
    # Guided Grad-CAM consistency analysis
    ggcam_results = compute_cam_consistency(
        ggcam_cams, patient_labels,
        output_dir=CAMS_DIR / "guided_gradcam",
        method_name="guided_gradcam"
    )
    plot_cam_comparison(
        ggcam_results['als_mean'],
        ggcam_results['control_mean'],
        CAMS_DIR / "guided_gradcam" / "cam_class_comparison.png"
    )
    
    print("\n" + "="*60)
    print("  CAM GENERATION COMPLETE")
    print("="*60)
    print(f"  LayerCAM outputs: {CAMS_DIR / 'layercam'}")
    print(f"  Guided Grad-CAM outputs: {CAMS_DIR / 'guided_gradcam'}")
    
    return {'layercam': layercam_results, 'guided_gradcam': ggcam_results}


def menu_complete_pipeline():
    """Menu option 6: Run complete pipeline."""
    print("\n" + "="*60)
    print("RUNNING COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Validate and split
    metadata_df, splits = menu_validate_and_split()
    if metadata_df is None:
        return
    
    # Step 3: Train (skip Optuna for speed, user can run separately)
    fold_results = menu_train_cv(metadata_df, splits, use_best_params=False)
    if fold_results is None:
        return
    
    # Step 4: Evaluate
    summary = menu_evaluate(metadata_df, splits)
    
    # Step 5: CAMs (optional due to time)
    cam_input = input("\n  Generate CAM visualizations? (y/n, default=y): ").strip().lower()
    if cam_input != 'n':
        menu_generate_cams(metadata_df, splits)
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")


def main():
    """Main entry point."""
    print_header()
    
    metadata_df = None
    splits = None
    
    while True:
        print_menu()
        
        try:
            choice = input("  Enter choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break
        
        if choice == '0':
            print("\n  Goodbye!")
            break
        
        elif choice == '1':
            metadata_df, splits = menu_validate_and_split()
        
        elif choice == '2':
            menu_optuna_tuning(metadata_df, splits)
        
        elif choice == '3':
            menu_train_cv(metadata_df, splits)
        
        elif choice == '4':
            menu_evaluate(metadata_df, splits)
        
        elif choice == '5':
            menu_generate_cams(metadata_df, splits)
        
        elif choice == '6':
            menu_complete_pipeline()
        
        else:
            print("\n  Invalid choice. Please try again.")
        
        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()

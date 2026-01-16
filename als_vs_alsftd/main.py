"""
Main Entry Point for ALS vs ALS-FTD LOPO Cartography Pipeline.

This script orchestrates the complete pipeline:
1. Load and filter data (Concordant = ALS-FTD, Discordant = ALS, exclude Controls)
2. Create LOPO patient-level splits
3. Train EfficientNetB0 across all folds and seeds
4. Compute windowed cartography metrics
5. Run reliability checks and patient categorization
6. Generate visualizations

Usage:
    python main.py
    
    # Or with specific stages:
    python main.py --stage splits     # Only create splits
    python main.py --stage train      # Only train (requires splits)
    python main.py --stage cartography # Only compute cartography (requires training logs)
    python main.py --stage reliability # Only run reliability checks
    python main.py --stage visualize  # Only generate visualizations
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

# Suppress albumentations version check warning
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from config import (
    DATA_DIR, IMAGE_DIR, IMAGE_KEYS_PATH,
    OUTPUT_DIR, LOPO_SPLITS_DIR, TRAINING_LOGS_DIR,
    CARTOGRAPHY_DIR, METRICS_DIR, FIGURES_DIR,
    SEEDS, POSITIVE_CATEGORY, NEGATIVE_CATEGORY
)
from utils import load_and_filter_als_vs_alsftd, set_seed
from lopo_splits import create_lopo_splits
from train import train_all_folds
from cartography import (
    compute_windowed_cartography_single_seed,
    aggregate_to_patient_level_single_seed,
    aggregate_across_seeds
)
from reliability_checks import run_reliability_checks, categorize_patients
from visualization import generate_all_visualizations


def print_banner():
    """Print pipeline banner."""
    print("\n" + "="*70)
    print("  ALS vs ALS-FTD Classification Pipeline")
    print("  LOPO Cross-Validation with Windowed Cartography")
    print("="*70)
    print(f"\n  Positive class (y=1): {POSITIVE_CATEGORY} → ALS-FTD")
    print(f"  Negative class (y=0): {NEGATIVE_CATEGORY} → ALS")
    print(f"  Seeds: {SEEDS}")
    print(f"  Output: {OUTPUT_DIR}\n")


def stage_1_create_splits():
    """Stage 1: Load data and create LOPO splits."""
    print("\n" + "="*70)
    print("STAGE 1: DATA LOADING AND LOPO SPLITS")
    print("="*70)
    
    # Load and filter data
    df = load_and_filter_als_vs_alsftd()
    
    # Create LOPO splits
    splits_df = create_lopo_splits(df)
    
    print(f"\n  ✓ Stage 1 complete!")
    print(f"  → {len(df)} images from {df['patient_id'].nunique()} patients")
    
    return df, splits_df


def stage_2_train(df):
    """Stage 2: Train model across all folds and seeds."""
    print("\n" + "="*70)
    print("STAGE 2: TRAINING")
    print("="*70)
    
    all_epoch_predictions = train_all_folds(df)
    
    print(f"\n  ✓ Stage 2 complete!")
    print(f"  → Training logs saved to: {TRAINING_LOGS_DIR}")
    
    return all_epoch_predictions


def stage_3_cartography(all_epoch_predictions, df):
    """Stage 3: Compute windowed cartography metrics."""
    print("\n" + "="*70)
    print("STAGE 3: WINDOWED CARTOGRAPHY")
    print("="*70)
    
    all_image_cartography = []
    all_patient_cartography = []
    
    for seed in SEEDS:
        print(f"\n  Processing seed {seed}...")
        
        # Filter predictions for this seed
        seed_predictions = {
            k: v for k, v in all_epoch_predictions.items()
            if k[1] == seed  # k = (fold, seed)
        }
        
        if not seed_predictions:
            print(f"    Warning: No predictions for seed {seed}")
            continue
        
        # Compute image-level cartography
        image_carto = compute_windowed_cartography_single_seed(seed_predictions, seed)
        all_image_cartography.append(image_carto)
        
        # Aggregate to patient level
        patient_carto = aggregate_to_patient_level_single_seed(image_carto, seed)
        all_patient_cartography.append(patient_carto)
    
    # Combine across seeds
    if all_image_cartography:
        combined_image = pd.concat(all_image_cartography, ignore_index=True)
    else:
        combined_image = pd.DataFrame()
    
    if all_patient_cartography:
        combined_patient = pd.concat(all_patient_cartography, ignore_index=True)
    else:
        combined_patient = pd.DataFrame()
    
    # Aggregate across seeds
    if len(combined_patient) > 0:
        patient_final = aggregate_across_seeds(combined_patient)
    else:
        patient_final = pd.DataFrame()
    
    # Save combined image cartography
    if len(combined_image) > 0:
        image_path = Path(CARTOGRAPHY_DIR) / "cartography_per_image_all_seeds.csv"
        combined_image.to_csv(image_path, index=False)
        print(f"\n  ✓ Saved image cartography: {image_path}")
    
    print(f"\n  ✓ Stage 3 complete!")
    
    return combined_image, patient_final


def stage_4_reliability(patient_final_df):
    """Stage 4: Run reliability checks."""
    print("\n" + "="*70)
    print("STAGE 4: RELIABILITY CHECKS")
    print("="*70)
    
    results = run_reliability_checks(patient_final_df)
    
    print(f"\n  ✓ Stage 4 complete!")
    print(f"  → Metrics saved to: {METRICS_DIR}")
    
    return results


def stage_5_visualize(patient_final_df, image_df=None, categorized_df=None):
    """Stage 5: Generate visualizations."""
    print("\n" + "="*70)
    print("STAGE 5: VISUALIZATION")
    print("="*70)
    
    generate_all_visualizations(patient_final_df, image_df, categorized_df)
    
    print(f"\n  ✓ Stage 5 complete!")
    print(f"  → Figures saved to: {FIGURES_DIR}")


def run_full_pipeline():
    """Run complete pipeline from start to finish."""
    print_banner()
    
    # Stage 1: Data and splits
    df, splits_df = stage_1_create_splits()
    
    # Stage 2: Training
    all_epoch_predictions = stage_2_train(df)
    
    # Stage 3: Cartography
    image_df, patient_final_df = stage_3_cartography(all_epoch_predictions, df)
    
    # Stage 4: Reliability checks
    results = stage_4_reliability(patient_final_df)
    categorized_df = results.get('categorization')
    
    # Stage 5: Visualizations
    stage_5_visualize(patient_final_df, image_df, categorized_df)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("\n  Key files:")
    print(f"    - LOPO splits:        {LOPO_SPLITS_DIR}/lopo_splits.csv")
    print(f"    - Cartography:        {CARTOGRAPHY_DIR}/cartography_per_patient_final.xlsx")
    print(f"    - Clinical metrics:   {METRICS_DIR}/lopo_clinical_metrics.csv")
    print(f"    - Categorization:     {METRICS_DIR}/patient_categorization.csv")
    print(f"    - Figures:            {FIGURES_DIR}/fig*.png")
    
    return patient_final_df, results


def load_existing_results():
    """Load results from previous run (for re-running specific stages)."""
    # Try to load patient final
    patient_final_path = Path(CARTOGRAPHY_DIR) / "cartography_per_patient_final.xlsx"
    if patient_final_path.exists():
        patient_final_df = pd.read_excel(patient_final_path)
    else:
        patient_final_df = None
    
    # Try to load image cartography
    image_path = Path(CARTOGRAPHY_DIR) / "cartography_per_image_all_seeds.csv"
    if image_path.exists():
        image_df = pd.read_csv(image_path)
    else:
        image_df = None
    
    # Try to load categorization
    cat_path = Path(METRICS_DIR) / "patient_categorization.csv"
    if cat_path.exists():
        categorized_df = pd.read_csv(cat_path)
    else:
        categorized_df = None
    
    return patient_final_df, image_df, categorized_df


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='ALS vs ALS-FTD LOPO Cartography Pipeline'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['full', 'splits', 'train', 'cartography', 'reliability', 'visualize'],
        default='full',
        help='Which stage to run (default: full pipeline)'
    )
    
    args = parser.parse_args()
    
    if args.stage == 'full':
        run_full_pipeline()
    
    elif args.stage == 'splits':
        print_banner()
        stage_1_create_splits()
    
    elif args.stage == 'train':
        print_banner()
        df = load_and_filter_als_vs_alsftd()
        stage_2_train(df)
    
    elif args.stage == 'cartography':
        print("\nNote: Cartography requires training epoch predictions.")
        print("Loading from saved training logs...")
        # Would need to implement loading from saved logs
        print("Please run full pipeline or implement log loading.")
    
    elif args.stage == 'reliability':
        print_banner()
        patient_final_df, _, _ = load_existing_results()
        if patient_final_df is not None:
            stage_4_reliability(patient_final_df)
        else:
            print("Error: No cartography results found. Run full pipeline first.")
    
    elif args.stage == 'visualize':
        print_banner()
        patient_final_df, image_df, categorized_df = load_existing_results()
        if patient_final_df is not None:
            stage_5_visualize(patient_final_df, image_df, categorized_df)
        else:
            print("Error: No results found. Run full pipeline first.")


if __name__ == '__main__':
    main()

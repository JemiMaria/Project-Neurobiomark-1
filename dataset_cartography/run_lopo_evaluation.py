"""
Run LOPO Evaluation - Main entry point for Leave-One-Patient-Out evaluation.

This script runs the complete LOPO evaluation pipeline:
1. 15 folds (one patient held out per fold)
2. 5 seeds per fold (ensemble stability)
3. Windowed metrics computation
4. Clinical metrics calculation
5. Visualization generation

Usage:
    python run_lopo_evaluation.py
    
Output:
    outputs/lopo/
    ├── lopo_per_image.xlsx          # Per-image windowed metrics
    ├── lopo_per_patient_seed.xlsx   # Per-patient-seed metrics
    ├── lopo_per_patient_final.xlsx  # Per-patient final metrics
    ├── lopo_clinical_metrics.csv    # Clinical evaluation metrics
    ├── lopo_training_logs.csv       # Training logs for all folds/seeds
    └── analysis/                    # Visualizations and README
        ├── lopo_patient_confidence_correctness.png
        ├── lopo_seed_stability.png
        ├── lopo_image_instability.png
        ├── lopo_metrics_across_folds.png
        ├── lopo_patient_predictions.png
        └── LOPO_README.md
"""

import sys
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import validate_paths, create_output_directories, LOPO_DIR, IMAGE_FOLDER, EXCEL_FILE, RANDOM_SEEDS, NUM_EPOCHS, WINDOW_HALF_SIZE
from lopo import run_lopo_evaluation


def main():
    """Run the complete LOPO evaluation pipeline."""
    
    print("=" * 70)
    print("    LOPO (Leave-One-Patient-Out) Evaluation")
    print("    Dataset Cartography for Medical Imaging")
    print("=" * 70)
    
    # Validate paths
    if not validate_paths():
        print("\nPlease update the paths in config.py before running.")
        return
    
    # Create output directories
    create_output_directories()
    
    print(f"\nConfiguration:")
    print(f"  - Image folder: {IMAGE_FOLDER}")
    print(f"  - Excel file: {EXCEL_FILE}")
    print(f"  - Output directory: {LOPO_DIR}")
    print(f"  - Seeds per fold: {len(RANDOM_SEEDS)}")
    print(f"  - Max epochs: {NUM_EPOCHS}")
    print(f"  - Window half-size: {WINDOW_HALF_SIZE}")
    
    print("\nThis will take a significant amount of time...")
    print("Estimated: ~15 folds × 5 seeds × 25 epochs = 1875 training iterations")
    print("-" * 70)
    
    start_time = time.time()
    
    # Run LOPO evaluation
    results = run_lopo_evaluation()
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("    LOPO Evaluation Complete!")
    print("=" * 70)
    print(f"\nTotal time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"\nOutput files saved to: {LOPO_DIR}")
    print("\nKey files:")
    print("  - lopo_per_patient_final.xlsx: Final patient-level metrics")
    print("  - lopo_clinical_metrics.csv: Clinical evaluation summary")
    print("  - analysis/LOPO_README.md: Detailed results summary")


if __name__ == "__main__":
    main()

"""
Dataset Cartography - Main Execution Script

This is the main entry point for running the dataset cartography analysis.
Run this script to:
1. Load and preprocess the data
2. Train models across multiple runs
3. Compute cartography metrics (both legacy and windowed)
4. Generate visualizations and report

Usage:
    python run_cartography.py
    
Before running, update the paths in config.py:
    - IMAGE_FOLDER: path to folder containing .tiff images
    - EXCEL_FILE: path to Excel file with metadata
"""

import os
import sys
import warnings
import time
from datetime import timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import all modules
from dataset_cartography.config import (
    create_output_directories, validate_paths,
    OUTPUT_DIR, LOGS_DIR, TRAINING_LOG_CSV, ANALYSIS_DIR
)
from dataset_cartography.data_loader import (
    load_metadata, split_data, 
    create_dataloaders, create_full_dataset_loader
)
from dataset_cartography.model import get_device
from dataset_cartography.trainer import run_all_training, save_training_logs
from dataset_cartography.cartography import (
    compute_cartography_metrics, save_metrics, print_summary
)
from dataset_cartography.windowed_cartography import (
    compute_windowed_cartography, save_windowed_cartography_results,
    validate_cartography_results, print_cartography_summary
)
from dataset_cartography.patient_analysis import run_all_analyses
from dataset_cartography.visualize import create_all_visualizations
from dataset_cartography.report import generate_html_report

import pandas as pd


def print_banner():
    """Print a welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              DATASET CARTOGRAPHY ANALYSIS                     ║
    ║                                                               ║
    ║     Analyzing Medical Imaging Dataset for Sample Difficulty   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """
    Main function to run the complete dataset cartography analysis.
    """
    start_time = time.time()
    
    print_banner()
    
    # =========================================================================
    # STEP 1: Validate Configuration
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Validating Configuration")
    print("="*60)
    
    if not validate_paths():
        print("\nPlease update the paths in config.py and run again.")
        return
    
    create_output_directories()
    print("✓ Configuration validated")
    
    # =========================================================================
    # STEP 2: Load Data
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Loading Data")
    print("="*60)
    
    # Load metadata from Excel
    metadata = load_metadata()
    print(f"✓ Loaded metadata for {len(metadata)} images")
    
    # Split into train/validation (using seed 42 for consistent splits)
    train_meta, val_meta = split_data(metadata, random_seed=42)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(train_meta, val_meta)
    full_loader = create_full_dataset_loader(metadata)
    print("✓ Dataloaders created")
    
    # =========================================================================
    # STEP 3: Get Device
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Setting Up Device")
    print("="*60)
    
    device = get_device()
    print(f"✓ Using device: {device}")
    
    # =========================================================================
    # STEP 4: Train Models (Multiple Runs)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: Training Models (5 runs with different seeds)")
    print("="*60)
    
    # Updated to receive new windowed cartography data
    (confidence_matrix, correctness_matrix, training_logs, 
     all_epoch_predictions, best_epochs) = run_all_training(
        train_loader, val_loader, full_loader, metadata, device=device
    )
    
    # Save training logs
    save_training_logs(training_logs, LOGS_DIR / TRAINING_LOG_CSV)
    print("✓ Training completed and logs saved")
    
    # =========================================================================
    # STEP 5: Compute Legacy Cartography Metrics
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: Computing Legacy Cartography Metrics")
    print("="*60)
    
    metrics_df = compute_cartography_metrics(
        confidence_matrix, correctness_matrix, metadata
    )
    
    # Save metrics
    save_metrics(metrics_df)
    print("✓ Legacy cartography metrics computed and saved")
    
    # Print summary
    print_summary(metrics_df)
    
    # =========================================================================
    # STEP 5b: Compute Windowed Cartography Metrics (NEW)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5b: Computing Windowed Cartography Metrics")
    print("="*60)
    
    # Compute windowed cartography at image and patient levels
    image_df, patient_df = compute_windowed_cartography(
        all_epoch_predictions, best_epochs, metadata
    )
    
    # Save windowed cartography results to Excel
    save_windowed_cartography_results(image_df, patient_df)
    
    # Validate results
    num_patients = metadata['patient_id'].nunique()
    validate_cartography_results(image_df, patient_df, expected_patients=num_patients)
    
    # Print summary
    print_cartography_summary(image_df, patient_df)
    
    print("✓ Windowed cartography metrics computed and saved")
    
    # =========================================================================
    # STEP 6: Create Visualizations
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: Creating Visualizations")
    print("="*60)
    
    create_all_visualizations(metrics_df)
    print("✓ Visualizations created")
    
    # =========================================================================
    # STEP 7: Generate Report
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 7: Generating HTML Report")
    print("="*60)
    
    training_logs_df = pd.DataFrame(training_logs)
    report_path = generate_html_report(metrics_df, training_logs_df)
    print("✓ HTML report generated")
    
    # =========================================================================
    # STEP 8: Patient-Level Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 8: Running Patient-Level Analysis")
    print("="*60)
    
    analysis_outputs = run_all_analyses()
    print("✓ Patient-level analysis completed")
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    elapsed_time = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nTotal time: {elapsed_str}")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nKey output files:")
    print(f"  - Legacy Metrics:     {OUTPUT_DIR / 'cartography_metrics.csv'}")
    print(f"  - Per-Image Table:    {OUTPUT_DIR / 'cartography_per_image.xlsx'}")
    print(f"  - Per-Patient Table:  {OUTPUT_DIR / 'cartography_per_patient.xlsx'}")
    print(f"  - Training Logs:      {LOGS_DIR / TRAINING_LOG_CSV}")
    print(f"  - Report:             {report_path}")
    print(f"  - Plots:              {OUTPUT_DIR / 'visualizations/'}")
    print(f"  - Patient Analysis:   {ANALYSIS_DIR}/")
    print("\nOpen the HTML report in a web browser to view the full analysis.")
    print("Check outputs/analysis/README.md for patient-level analysis summary.")



if __name__ == "__main__":
    main()

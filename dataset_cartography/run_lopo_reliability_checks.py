"""
Run LOPO Reliability Checks - Standalone script for reliability analysis.

This script performs reliability checks on existing LOPO results:
A) Global clinical metrics with 95% Wilson confidence intervals
B) Probability margin analysis (class separation)
C) Seed variance analysis (prediction instability)
D) Hardest patient identification

Prerequisites:
    Run LOPO evaluation first: python run_lopo_evaluation.py
    
Input files (auto-detected from outputs/lopo/):
    - lopo_per_patient_final.xlsx
    - lopo_per_patient_seed.xlsx

Usage:
    python run_lopo_reliability_checks.py
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lopo.lopo_reliability_checks import run_reliability_checks


if __name__ == "__main__":
    run_reliability_checks()

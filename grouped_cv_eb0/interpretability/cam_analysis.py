"""
CAM consistency analysis across patients and classes.

Computes:
- Within-patient consistency (IoU of top-activated regions)
- Patient-to-class similarity (correlation with class mean CAM)
- Between-class separability
- Outlier detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAMS_DIR, CAM_TOP_PERCENTILE


def compute_top_percentile_mask(cam, percentile=None):
    """
    Get binary mask of top percentile activated pixels.
    
    Args:
        cam: CAM array [H, W]
        percentile: Top percentile to include (default from config)
        
    Returns:
        np.ndarray: Binary mask
    """
    if percentile is None:
        percentile = CAM_TOP_PERCENTILE
    
    threshold = np.percentile(cam, 100 - percentile)
    return (cam >= threshold).astype(int)


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two masks.
    
    Args:
        mask1, mask2: Binary masks
        
    Returns:
        float: IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_dice(mask1, mask2):
    """
    Compute Dice coefficient between two masks.
    
    Args:
        mask1, mask2: Binary masks
        
    Returns:
        float: Dice score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    
    if sum_masks == 0:
        return 0.0
    
    return 2 * intersection / sum_masks


def compute_within_patient_consistency(patient_cams, percentile=None):
    """
    Compute consistency of CAM activations within each patient.
    
    For each patient, computes IoU/Dice between all pairs of their images'
    top-activated regions.
    
    Args:
        patient_cams: Dict {patient_id: list of CAM arrays}
        percentile: Top percentile for mask
        
    Returns:
        pd.DataFrame: Consistency metrics per patient
    """
    records = []
    
    for pid, cams in patient_cams.items():
        if len(cams) < 2:
            records.append({
                'patient_id': pid,
                'n_images': len(cams),
                'mean_iou': np.nan,
                'std_iou': np.nan,
                'mean_dice': np.nan,
                'std_dice': np.nan
            })
            continue
        
        # Get masks for all images
        masks = [compute_top_percentile_mask(cam, percentile) for cam in cams]
        
        # Compute pairwise IoU and Dice
        ious = []
        dices = []
        
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                ious.append(compute_iou(masks[i], masks[j]))
                dices.append(compute_dice(masks[i], masks[j]))
        
        records.append({
            'patient_id': pid,
            'n_images': len(cams),
            'mean_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'mean_dice': np.mean(dices),
            'std_dice': np.std(dices)
        })
    
    return pd.DataFrame(records)


def compute_class_mean_cams(patient_cams, patient_labels):
    """
    Compute mean CAM for each class.
    
    Args:
        patient_cams: Dict {patient_id: list of CAM arrays}
        patient_labels: Dict {patient_id: y_true}
        
    Returns:
        tuple: (als_mean_cam, control_mean_cam)
    """
    als_cams = []
    control_cams = []
    
    for pid, cams in patient_cams.items():
        label = patient_labels.get(pid, None)
        if label is None:
            continue
        
        # Average this patient's CAMs
        patient_avg = np.mean(cams, axis=0)
        
        if label == 1:
            als_cams.append(patient_avg)
        else:
            control_cams.append(patient_avg)
    
    # Compute class means
    als_mean = np.mean(als_cams, axis=0) if als_cams else None
    control_mean = np.mean(control_cams, axis=0) if control_cams else None
    
    return als_mean, control_mean


def compute_patient_to_class_similarity(patient_cams, patient_labels, als_mean, control_mean):
    """
    Compute similarity between each patient's CAM and their class mean.
    
    Uses Pearson correlation.
    
    Args:
        patient_cams: Dict {patient_id: list of CAM arrays}
        patient_labels: Dict {patient_id: y_true}
        als_mean: ALS class mean CAM
        control_mean: Control class mean CAM
        
    Returns:
        pd.DataFrame: Similarity metrics per patient
    """
    records = []
    
    for pid, cams in patient_cams.items():
        label = patient_labels.get(pid, None)
        if label is None:
            continue
        
        patient_avg = np.mean(cams, axis=0).flatten()
        
        # Get appropriate class mean
        class_mean = (als_mean if label == 1 else control_mean).flatten()
        
        # Compute correlation
        corr, p_value = stats.pearsonr(patient_avg, class_mean)
        
        records.append({
            'patient_id': pid,
            'y_true': label,
            'correlation_to_class_mean': corr,
            'p_value': p_value
        })
    
    return pd.DataFrame(records)


def compute_between_class_separability(als_mean, control_mean):
    """
    Compute similarity between ALS and Control mean CAMs.
    
    Lower similarity = better separability.
    
    Args:
        als_mean: ALS class mean CAM
        control_mean: Control class mean CAM
        
    Returns:
        dict: Separability metrics
    """
    if als_mean is None or control_mean is None:
        return {'correlation': np.nan, 'iou': np.nan}
    
    # Correlation
    corr, _ = stats.pearsonr(als_mean.flatten(), control_mean.flatten())
    
    # IoU of top regions
    als_mask = compute_top_percentile_mask(als_mean)
    control_mask = compute_top_percentile_mask(control_mean)
    iou = compute_iou(als_mask, control_mask)
    
    return {
        'correlation': corr,
        'iou': iou
    }


def compute_cam_outliers(consistency_df, similarity_df, consistency_threshold=0.1, similarity_percentile=10):
    """
    Identify CAM outlier patients.
    
    Outliers:
    - Low within-patient consistency (IoU below threshold)
    - Low patient-to-class similarity (bottom percentile)
    
    Args:
        consistency_df: DataFrame from compute_within_patient_consistency
        similarity_df: DataFrame from compute_patient_to_class_similarity
        consistency_threshold: IoU threshold for low consistency
        similarity_percentile: Percentile for low similarity
        
    Returns:
        pd.DataFrame: Outlier patients with reasons
    """
    outliers = []
    
    # Merge dataframes
    merged = consistency_df.merge(similarity_df, on='patient_id', how='outer')
    
    # Compute similarity threshold
    sim_threshold = np.nanpercentile(merged['correlation_to_class_mean'], similarity_percentile)
    
    for _, row in merged.iterrows():
        reasons = []
        
        if not np.isnan(row.get('mean_iou', np.nan)) and row['mean_iou'] < consistency_threshold:
            reasons.append(f"low_consistency (IoU={row['mean_iou']:.3f})")
        
        if not np.isnan(row.get('correlation_to_class_mean', np.nan)) and row['correlation_to_class_mean'] < sim_threshold:
            reasons.append(f"low_class_similarity (r={row['correlation_to_class_mean']:.3f})")
        
        if reasons:
            outliers.append({
                'patient_id': row['patient_id'],
                'y_true': row.get('y_true', np.nan),
                'reasons': '; '.join(reasons),
                'mean_iou': row.get('mean_iou', np.nan),
                'correlation_to_class_mean': row.get('correlation_to_class_mean', np.nan)
            })
    
    return pd.DataFrame(outliers)


def compute_cam_consistency(patient_cams, patient_labels, output_dir=None, method_name="cam"):
    """
    Run full CAM consistency analysis.
    
    Args:
        patient_cams: Dict {patient_id: list of CAM arrays}
        patient_labels: Dict {patient_id: y_true}
        output_dir: Output directory for this method
        method_name: Name of CAM method
        
    Returns:
        dict: All analysis results
    """
    if output_dir is None:
        output_dir = CAMS_DIR / method_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CAM CONSISTENCY ANALYSIS: {method_name.upper()}")
    print(f"{'='*60}")
    
    # Within-patient consistency
    print("\n  Computing within-patient consistency...")
    consistency_df = compute_within_patient_consistency(patient_cams)
    consistency_df.to_csv(output_dir / "cam_patient_consistency.csv", index=False)
    
    # Class mean CAMs
    print("  Computing class mean CAMs...")
    als_mean, control_mean = compute_class_mean_cams(patient_cams, patient_labels)
    
    # Patient-to-class similarity
    print("  Computing patient-to-class similarity...")
    similarity_df = compute_patient_to_class_similarity(
        patient_cams, patient_labels, als_mean, control_mean
    )
    similarity_df.to_csv(output_dir / "cam_patient_similarity.csv", index=False)
    
    # Between-class separability
    separability = compute_between_class_separability(als_mean, control_mean)
    print(f"  Between-class separability: corr={separability['correlation']:.3f}, IoU={separability['iou']:.3f}")
    
    # Outliers
    print("  Identifying outliers...")
    outliers_df = compute_cam_outliers(consistency_df, similarity_df)
    outliers_df.to_csv(output_dir / "cam_outlier_patients.csv", index=False)
    
    # Summary
    print(f"\n  Results:")
    print(f"    Mean within-patient IoU: {consistency_df['mean_iou'].mean():.3f} ± {consistency_df['mean_iou'].std():.3f}")
    print(f"    Mean patient-to-class correlation: {similarity_df['correlation_to_class_mean'].mean():.3f}")
    print(f"    Outlier patients: {len(outliers_df)}")
    
    print(f"\n  ✓ Saved: cam_patient_consistency.csv")
    print(f"  ✓ Saved: cam_patient_similarity.csv")
    print(f"  ✓ Saved: cam_outlier_patients.csv")
    
    return {
        'consistency': consistency_df,
        'similarity': similarity_df,
        'separability': separability,
        'outliers': outliers_df,
        'als_mean': als_mean,
        'control_mean': control_mean
    }


def plot_cam_comparison(als_mean, control_mean, output_path):
    """
    Plot class comparison: ALS mean, Control mean, and difference.
    
    Args:
        als_mean: ALS class mean CAM
        control_mean: Control class mean CAM
        output_path: Path to save plot
    """
    if als_mean is None or control_mean is None:
        print("  ⚠ Cannot plot comparison: missing class mean CAMs")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # ALS mean
    im0 = axes[0].imshow(als_mean, cmap='jet')
    axes[0].set_title("ALS Mean CAM")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Control mean
    im1 = axes[1].imshow(control_mean, cmap='jet')
    axes[1].set_title("Control Mean CAM")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Difference (ALS - Control)
    diff = als_mean - control_mean
    max_abs = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title("Difference (ALS - Control)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Absolute difference
    im3 = axes[3].imshow(np.abs(diff), cmap='hot')
    axes[3].set_title("|ALS - Control|")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save difference map separately
    diff_path = output_path.parent / "cam_difference_map.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    ax.set_title("CAM Difference: ALS - Control", fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='ALS higher ← → Control higher')
    plt.tight_layout()
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Saved: {diff_path}")

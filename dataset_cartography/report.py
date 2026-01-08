"""
Report generation for Dataset Cartography.

This module generates a comprehensive HTML report with:
- Explanation of dataset cartography
- Category distribution and metrics
- Per-patient breakdown
- Embedded visualizations
- Key findings and recommendations
"""

import os
import base64
from datetime import datetime
import pandas as pd

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    OUTPUT_DIR, VISUALIZATIONS_DIR, DOCUMENTATION_FILE,
    SCATTER_PLOT_PNG, CATEGORY_DISTRIBUTION_PNG,
    NUM_RUNS, RANDOM_SEEDS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
)
from dataset_cartography.cartography import get_category_distribution, get_per_patient_breakdown


def image_to_base64(image_path):
    """
    Convert an image file to base64 string for embedding in HTML.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string or None if file not found
    """
    if not os.path.exists(image_path):
        return None
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def generate_html_report(metrics_df, training_logs_df=None, output_path=None):
    """
    Generate comprehensive HTML report.
    
    Args:
        metrics_df: DataFrame with cartography metrics
        training_logs_df: DataFrame with training logs (optional)
        output_path: Path to save the report
        
    Returns:
        Path to generated report
    """
    if output_path is None:
        output_path = OUTPUT_DIR / DOCUMENTATION_FILE
    
    # Get category distribution
    distribution = get_category_distribution(metrics_df)
    total_samples = sum(distribution.values())
    
    # Get per-patient breakdown
    patient_breakdown = get_per_patient_breakdown(metrics_df)
    
    # Load visualizations as base64
    scatter_path = VISUALIZATIONS_DIR / SCATTER_PLOT_PNG
    category_path = VISUALIZATIONS_DIR / CATEGORY_DISTRIBUTION_PNG
    sample_images_path = VISUALIZATIONS_DIR / "sample_images.png"
    
    scatter_b64 = image_to_base64(scatter_path)
    category_b64 = image_to_base64(category_path)
    sample_b64 = image_to_base64(sample_images_path)
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Cartography Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .summary-box {{
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
        }}
        .category-easy {{ color: #27ae60; font-weight: bold; }}
        .category-medium {{ color: #3498db; font-weight: bold; }}
        .category-ambiguous {{ color: #f39c12; font-weight: bold; }}
        .category-hard {{ color: #e74c3c; font-weight: bold; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        .visualization img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: white;
        }}
        .stat-card.easy {{ background-color: #27ae60; }}
        .stat-card.medium {{ background-color: #3498db; }}
        .stat-card.ambiguous {{ background-color: #f39c12; }}
        .stat-card.hard {{ background-color: #e74c3c; }}
        .stat-number {{
            font-size: 36px;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .config-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 14px;
        }}
        .recommendation {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 5px solid #ffc107;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Dataset Cartography Analysis Report</h1>
        <p><em>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
        
        <h2>📖 What is Dataset Cartography?</h2>
        <div class="summary-box">
            <p><strong>Dataset Cartography</strong> is a technique for characterizing and diagnosing datasets 
            based on how a model learns each sample over training. By tracking confidence and correctness 
            across multiple training epochs and runs, we can identify which samples are:</p>
            <ul>
                <li><span class="category-easy">Easy</span>: Consistently predicted correctly with high confidence</li>
                <li><span class="category-medium">Medium</span>: Moderately difficult samples</li>
                <li><span class="category-ambiguous">Ambiguous</span>: Inconsistently predicted (sometimes right, sometimes wrong)</li>
                <li><span class="category-hard">Hard</span>: Consistently predicted incorrectly - may indicate mislabeled data or outliers</li>
            </ul>
        </div>
        
        <h2>📊 Overall Results</h2>
        
        <div class="stat-grid">
            <div class="stat-card easy">
                <div class="stat-number">{distribution['Easy']}</div>
                <div class="stat-label">Easy Samples<br>({100*distribution['Easy']/total_samples:.1f}%)</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-number">{distribution['Medium']}</div>
                <div class="stat-label">Medium Samples<br>({100*distribution['Medium']/total_samples:.1f}%)</div>
            </div>
            <div class="stat-card ambiguous">
                <div class="stat-number">{distribution['Ambiguous']}</div>
                <div class="stat-label">Ambiguous Samples<br>({100*distribution['Ambiguous']/total_samples:.1f}%)</div>
            </div>
            <div class="stat-card hard">
                <div class="stat-number">{distribution['Hard']}</div>
                <div class="stat-label">Hard Samples<br>({100*distribution['Hard']/total_samples:.1f}%)</div>
            </div>
        </div>
        
        <h3>Metrics Summary</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
            <tr>
                <td>Confidence</td>
                <td>{metrics_df['confidence'].mean():.3f}</td>
                <td>{metrics_df['confidence'].std():.3f}</td>
                <td>{metrics_df['confidence'].min():.3f}</td>
                <td>{metrics_df['confidence'].max():.3f}</td>
            </tr>
            <tr>
                <td>Variability</td>
                <td>{metrics_df['variability'].mean():.3f}</td>
                <td>{metrics_df['variability'].std():.3f}</td>
                <td>{metrics_df['variability'].min():.3f}</td>
                <td>{metrics_df['variability'].max():.3f}</td>
            </tr>
            <tr>
                <td>Correctness</td>
                <td>{metrics_df['correctness'].mean():.3f}</td>
                <td>{metrics_df['correctness'].std():.3f}</td>
                <td>{metrics_df['correctness'].min():.3f}</td>
                <td>{metrics_df['correctness'].max():.3f}</td>
            </tr>
        </table>
        
        <h2>📈 Visualizations</h2>
        
        {'<div class="visualization"><h3>Dataset Cartography Scatter Plot</h3><img src="data:image/png;base64,' + scatter_b64 + '" alt="Cartography Scatter Plot"></div>' if scatter_b64 else '<p>Scatter plot not available.</p>'}
        
        {'<div class="visualization"><h3>Category Distribution</h3><img src="data:image/png;base64,' + category_b64 + '" alt="Category Distribution"></div>' if category_b64 else '<p>Category distribution plot not available.</p>'}
        
        {'<div class="visualization"><h3>Sample Images from Each Category</h3><img src="data:image/png;base64,' + sample_b64 + '" alt="Sample Images"></div>' if sample_b64 else ''}
        
        <h2>👥 Per-Patient Breakdown</h2>
        <p>This table shows how samples from each patient are distributed across categories.</p>
        
        <table>
            <tr>
                <th>Patient ID</th>
                <th>Easy</th>
                <th>Medium</th>
                <th>Ambiguous</th>
                <th>Hard</th>
                <th>Total</th>
                <th>Hard+Ambiguous %</th>
            </tr>
"""
    
    # Add patient breakdown rows
    for _, row in patient_breakdown.iterrows():
        hard_amb_pct = 100 * row['Hard_Ambiguous_Ratio']
        html_content += f"""
            <tr>
                <td>{row['patient_id']}</td>
                <td>{int(row['Easy'])}</td>
                <td>{int(row['Medium'])}</td>
                <td>{int(row['Ambiguous'])}</td>
                <td>{int(row['Hard'])}</td>
                <td>{int(row['Total'])}</td>
                <td style="{'color: red; font-weight: bold;' if hard_amb_pct > 50 else ''}">{hard_amb_pct:.1f}%</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h2>💡 Key Findings & Recommendations</h2>
"""
    
    # Generate findings based on data
    easy_pct = 100 * distribution['Easy'] / total_samples
    hard_pct = 100 * distribution['Hard'] / total_samples
    ambiguous_pct = 100 * distribution['Ambiguous'] / total_samples
    
    findings = []
    
    if easy_pct > 60:
        findings.append("✅ Most samples are easy to learn, suggesting good data quality for the majority of the dataset.")
    
    if hard_pct > 20:
        findings.append("⚠️ Significant portion of hard samples detected. Consider reviewing these for potential labeling errors or quality issues.")
    
    if ambiguous_pct > 15:
        findings.append("🔍 Notable number of ambiguous samples. These may represent borderline cases or samples near the decision boundary.")
    
    # Find problematic patients
    high_risk_patients = patient_breakdown[patient_breakdown['Hard_Ambiguous_Ratio'] > 0.5]
    if len(high_risk_patients) > 0:
        patient_list = ", ".join(high_risk_patients['patient_id'].tolist()[:5])
        findings.append(f"👤 Patients with >50% hard/ambiguous samples: {patient_list}")
    
    # Add condition-specific findings
    for condition in metrics_df['condition'].unique():
        cond_df = metrics_df[metrics_df['condition'] == condition]
        cond_hard_pct = 100 * (cond_df['category'] == 'Hard').sum() / len(cond_df)
        if cond_hard_pct > 25:
            findings.append(f"⚠️ {condition} samples have {cond_hard_pct:.1f}% hard samples - may need special attention.")
    
    if not findings:
        findings.append("📊 Dataset shows balanced distribution across categories with no major concerns.")
    
    for finding in findings:
        html_content += f"""
        <div class="recommendation">
            {finding}
        </div>
"""
    
    html_content += f"""
        <h2>⚙️ Training Configuration</h2>
        <div class="config-box">
            <strong>Model:</strong> EfficientNetB0 (pretrained on ImageNet)<br>
            <strong>Training Runs:</strong> {NUM_RUNS} runs with seeds {RANDOM_SEEDS}<br>
            <strong>Epochs:</strong> {NUM_EPOCHS} (with early stopping, patience=5)<br>
            <strong>Batch Size:</strong> {BATCH_SIZE}<br>
            <strong>Learning Rate:</strong> {LEARNING_RATE}<br>
            <strong>Optimizer:</strong> AdamW (weight_decay=1e-4)<br>
            <strong>Loss:</strong> BCEWithLogitsLoss (pos_weight=0.583)<br>
            <strong>Image Size:</strong> 224x224 with ImageNet normalization<br>
            <strong>Dropout:</strong> 0.3 before final classifier
        </div>
        
        <h2>📂 Output Files</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>cartography_metrics.csv</td>
                <td>All metrics for each sample (confidence, variability, correctness, category)</td>
            </tr>
            <tr>
                <td>training_logs.csv</td>
                <td>Training/validation loss and accuracy for each epoch and run</td>
            </tr>
            <tr>
                <td>cartography_scatter_plot.png</td>
                <td>Main visualization showing confidence vs variability</td>
            </tr>
            <tr>
                <td>category_distribution.png</td>
                <td>Bar chart showing sample distribution across categories</td>
            </tr>
        </table>
        
        <h2>🔍 How to Use These Results</h2>
        <ol>
            <li><strong>Review Hard Samples:</strong> Manually inspect images categorized as "Hard" - they may have labeling errors or quality issues.</li>
            <li><strong>Examine Ambiguous Samples:</strong> These represent borderline cases that may need expert review or additional annotation.</li>
            <li><strong>Patient-Level Analysis:</strong> Investigate patients with high proportions of hard/ambiguous samples - may indicate systematic issues.</li>
            <li><strong>Data Cleaning:</strong> Consider removing or relabeling consistently mispredicted samples before final model training.</li>
            <li><strong>Model Improvement:</strong> Use easy samples for initial training and gradually incorporate harder samples (curriculum learning).</li>
        </ol>
        
        <div class="footer">
            <p>Dataset Cartography Analysis | Generated using PyTorch and EfficientNetB0</p>
            <p>Reference: Swayamdipta et al., "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save the report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("Report generation module loaded successfully.")
    print("Use generate_html_report(metrics_df) after completing analysis.")

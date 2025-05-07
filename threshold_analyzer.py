import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report
)
import json

def load_predictions(file_path):
    """Load predictions CSV with required columns"""
    df = pd.read_csv(file_path)
    required_cols = {'actual_wicket', 'wicket_probability'}
    assert required_cols.issubset(df.columns), \
        f"CSV must contain {required_cols} columns"
    return df

def process_threshold(df, threshold, output_dir):
    """Save predictions for a specific threshold"""
    # Create threshold-specific directory
    thresh_dir = os.path.join(output_dir, f"thresh_{threshold:.2f}")
    os.makedirs(thresh_dir, exist_ok=True)
    
    # Add predictions for this threshold
    df_thresh = df.copy()
    df_thresh['predicted_wicket'] = (df_thresh['wicket_probability'] >= threshold).astype(int)
    
    # Save predictions
    pred_path = os.path.join(thresh_dir, 'predictions.csv')
    df_thresh.to_csv(pred_path, index=False)
    
    return df_thresh['predicted_wicket']

def analyze_thresholds(df, thresholds, output_dir):
    """Analyze and save results for multiple thresholds"""
    y_true = df['actual_wicket']
    y_proba = df['wicket_probability']
    
    results = []
    pr_data = []
    
    # Threshold-independent metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    
    for thresh in thresholds:
        # Save predictions and get y_pred
        y_pred = process_threshold(df, thresh, output_dir)
        
        # Calculate metrics
        res = {
            'threshold': thresh,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'alerts': y_pred.sum(),
            'true_positives': (y_pred & y_true).sum(),
            'false_positives': (y_pred & ~y_true).sum(),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        results.append(res)
    
    return pd.DataFrame(results), (precision, recall, pr_thresholds)

def save_summary(results_df, pr_data, output_dir):
    """Save analysis summary and PR curve data"""
    # Save numerical results
    results_path = os.path.join(output_dir, 'threshold_analysis.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save PR curve data
    precision, recall, thresholds = pr_data
    pr_df = pd.DataFrame({
        'threshold': np.append(thresholds, np.nan),
        'precision': precision,
        'recall': recall
    })
    pr_df.to_csv(os.path.join(output_dir, 'precision_recall_curve.csv'), index=False)
    
    # Save JSON summary
    best_idx = results_df['f1'].idxmax()
    summary = {
        'best_threshold': results_df.loc[best_idx, 'threshold'],
        'best_f1': results_df.loc[best_idx, 'f1'],
        'best_precision': results_df.loc[best_idx, 'precision'],
        'best_recall': results_df.loc[best_idx, 'recall'],
        'roc_auc': results_df['roc_auc'].iloc[0],
        'pr_auc': results_df['pr_auc'].iloc[0]
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Threshold Analysis with Prediction Saving')
    parser.add_argument('--input', required=True, help='Path to predictions CSV')
    parser.add_argument('--output_dir', default='threshold_analysis',
                       help='Output directory for results')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                       help='Thresholds to evaluate')
    
    args = parser.parse_args()
    
    # Load data
    df = load_predictions(args.input)
    
    # Process thresholds
    results_df, pr_data = analyze_thresholds(df, args.thresholds, args.output_dir)
    
    # Save summary results
    save_summary(results_df, pr_data, args.output_dir)
    
    # Print summary
    print("\nAnalysis Complete")
    print(f"Results saved to {args.output_dir}")
    print("\nThreshold Performance Summary:")
    print(results_df[['threshold', 'precision', 'recall', 'f1', 'alerts']].to_markdown(index=False))

if __name__ == '__main__':
    main()
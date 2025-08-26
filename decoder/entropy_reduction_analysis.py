"""
Entropy Reduction Analysis - Ambiguity-Reduction Score (ARS) Calculations

This script computes the Ambiguity-Reduction Score (ARS) for both classification
and regression tasks, providing a measure of how much ambiguity has been eliminated
from neural representations through training.

The ARS is defined as: ARS = 1 - H(I|R,C) / H_max
where H(I|R,C) is the conditional entropy of interpretations given representation R
and context C, and H_max is the maximum possible entropy.

For classification: Uses Fano's inequality to provide lower bounds
For regression: Uses R² scores assuming Gaussian residuals
"""

import math
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List


def binary_entropy(p: float) -> float:
    """
    Calculate binary entropy h_b(p) = -p*log2(p) - (1-p)*log2(1-p)
    
    Args:
        p: Probability value between 0 and 1
        
    Returns:
        Binary entropy value
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def ars_classification_lower_bound(accuracy: float, num_classes: int = 10) -> float:
    """
    Calculate lower bound for ARS using Fano's inequality for classification.
    
    Based on: ARS ≥ 1 - [h_b(1-A) + (1-A)*log2(K-1)] / log2(K)
    where A is accuracy, K is number of classes, and h_b is binary entropy.
    
    Args:
        accuracy: Top-1 accuracy (between 0 and 1)
        num_classes: Number of classes (K)
        
    Returns:
        Lower bound estimate of ARS
    """
    if accuracy < 0 or accuracy > 1:
        raise ValueError("Accuracy must be between 0 and 1")
    if num_classes < 2:
        raise ValueError("Number of classes must be at least 2")
    
    error_rate = 1.0 - accuracy
    h_b_error = binary_entropy(error_rate)
    
    numerator = h_b_error + error_rate * math.log2(num_classes - 1)
    denominator = math.log2(num_classes)
    
    ars = 1.0 - (numerator / denominator)
    return max(0.0, min(1.0, ars))  # Clamp to [0, 1]


def ars_regression_lower_bound(r2_score: float) -> float:
    """
    Calculate lower bound for ARS using R² score for regression tasks.
    
    Based on: ARS ≥ log2[1/(1-R²)] / log2(2πe)
    Assumes Gaussian residuals and standardized target variance = 1.
    
    Args:
        r2_score: R² score (coefficient of determination)
        
    Returns:
        Lower bound estimate of ARS
    """
    if r2_score >= 1.0:
        return 1.0
    if r2_score <= 0.0:
        return 0.0
    
    numerator = math.log2(1.0 / (1.0 - r2_score))
    denominator = math.log2(2 * math.pi * math.e)  # ≈ 4.094
    
    ars = numerator / denominator
    return max(0.0, min(1.0, ars))  # Clamp to [0, 1]


def load_classification_results(file_path: str) -> pd.DataFrame:
    """
    Load and process classification results for ARS calculation.
    
    Args:
        file_path: Path to CSV file with classification results
        
    Returns:
        DataFrame with processed results including ARS lower bounds
    """
    df = pd.read_csv(file_path)
    
    # Label experiment conditions
    df["condition"] = np.select(
        [df["untrained"], df["model_class_str"] == "fully_connected_dropout"],
        ["untrained", "dropout"],
        default="no_dropout"
    )
    
    # Compute ARS lower bound per run
    df["ARS_lb"] = df["valid_acc"].apply(ars_classification_lower_bound)
    
    return df


def summarize_classification_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize classification results by condition.
    
    Args:
        df: DataFrame with classification results
        
    Returns:
        Summary DataFrame with mean and std for accuracy and ARS
    """
    summary = (
        df.groupby("condition")
        .agg(
            acc_mean=('valid_acc', 'mean'),
            acc_std=('valid_acc', 'std'),
            ars_mean=('ARS_lb', 'mean'),
            ars_std=('ARS_lb', 'std'),
        )
        .round(3)
        .reset_index()
    )
    
    return summary


def calculate_distance_from_center_targets() -> np.ndarray:
    """
    Calculate normalized distance from center for 28x28 MNIST pixels.
    
    Returns:
        Array of normalized distances from center for each pixel
    """
    distances = []
    for i in range(28):
        for j in range(28):
            dist = math.sqrt((i - 13.5)**2 + (j - 13.5)**2)
            max_dist = math.sqrt(13.5**2 + 13.5**2)
            distances.append(dist / max_dist)
    
    return np.array(distances)


def load_input_pixel_results(
    data_sources: list, 
    targets: np.ndarray,
    data_dir: str = "data/input-pixels"
) -> Dict[str, pd.DataFrame]:
    """
    Load input pixel decoding results and calculate R² scores.
    
    Args:
        data_sources: List of data source identifiers
        targets: Target values for computing label variance
        data_dir: Directory containing the data files
        
    Returns:
        Dictionary mapping labels to DataFrames with R² scores
    """
    column_names = ['step', 'mean_mse', 'max_mse', 'min_mse']
    label_variance = np.var(targets)
    
    results = {}
    for source in data_sources:
        file_path = f"{data_dir}/{source}.csv"
        label = source.replace('-y', '')
        
        try:
            df = pd.read_csv(file_path)
            df.columns = column_names
            
            # Calculate R² scores
            df['mean_r2'] = 1.0 - (df['mean_mse'] / label_variance)
            df['min_r2'] = 1.0 - (df['max_mse'] / label_variance)
            df['max_r2'] = 1.0 - (df['min_mse'] / label_variance)
            
            results[label] = df
            
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping: {source}")
    
    return results


def analyze_regression_ars(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze regression results and calculate ARS lower bounds.
    
    Args:
        results_dict: Dictionary with regression results
        
    Returns:
        DataFrame with R² values and ARS lower bounds
    """
    final_r2_values = {}
    for label, df in results_dict.items():
        final_r2_values[label] = df['mean_r2'].values[-1]
    
    # Create summary DataFrame
    df = (pd.Series(final_r2_values, name="R2")
          .reset_index()
          .rename(columns={"index": "condition"}))
    
    df["ARS_lb"] = df["R2"].apply(ars_regression_lower_bound).round(3)
    
    return df


def print_classification_summary(summary_df: pd.DataFrame) -> None:
    """Print formatted classification results summary."""
    print("=== CLASSIFICATION ARS ANALYSIS ===")
    print("Condition     | Accuracy (μ±σ)  | ARS Lower Bound (μ±σ)")
    print("-" * 55)
    
    for _, row in summary_df.iterrows():
        condition = row['condition']
        acc_mean, acc_std = row['acc_mean'], row['acc_std']
        ars_mean, ars_std = row['ars_mean'], row['ars_std']
        
        print(f"{condition:12s} | {acc_mean:.3f}±{acc_std:.3f}    | {ars_mean:.3f}±{ars_std:.3f}")


def print_regression_summary(summary_df: pd.DataFrame) -> None:
    """Print formatted regression results summary."""
    print("\n=== REGRESSION ARS ANALYSIS ===")
    print("Condition     | R² Score    | ARS Lower Bound")
    print("-" * 45)
    
    for _, row in summary_df.iterrows():
        condition = row['condition']
        r2 = row['R2']
        ars = row['ARS_lb']
        
        print(f"{condition:12s} | {r2:.3f}       | {ars:.3f}")


def run_gram_matrix_comparison() -> Tuple[List[float], List[float], List[str]]:
    """
    Run the gram matrix decoder comparison experiment.
    
    Returns:
        Tuple of (accuracies, accuracy_stds, model_names)
    """
    try:
        # Add gram_matrix_decoder to path
        gram_matrix_path = Path(__file__).parent.parent / "gram_matrix_decoder"
        sys.path.insert(0, str(gram_matrix_path))
        
        from runs.classid_comparison import run_comparison_experiment
        
        # Run the experiment and get results
        accuracies, accuracy_stds = run_comparison_experiment()
        
        # The experiment returns results in the order: untrained, no_dropout, dropout
        model_names = ['untrained', 'no_dropout', 'dropout']
        
        return accuracies, accuracy_stds, model_names
        
    except ImportError as e:
        print(f"Error importing gram matrix decoder: {e}")
        return [], [], []
    except Exception as e:
        print(f"Error running gram matrix experiment: {e}")
        return [], [], []


def analyze_gram_matrix_ars(accuracies: List[float], model_names: List[str]) -> pd.DataFrame:
    """
    Analyze gram matrix decoder results and calculate ARS lower bounds.
    
    Args:
        accuracies: List of position accuracies
        model_names: List of model condition names
        
    Returns:
        DataFrame with accuracy values and ARS lower bounds
    """
    # Create DataFrame
    df = pd.DataFrame({
        'condition': model_names,
        'position_accuracy': accuracies
    })
    
    # Calculate ARS lower bounds (using 10 classes for MNIST digits)
    df["ARS_lb"] = df["position_accuracy"].apply(
        lambda x: ars_classification_lower_bound(x, num_classes=10)
    ).round(3)
    
    return df


def print_gram_matrix_summary(summary_df: pd.DataFrame) -> None:
    """Print formatted gram matrix decoder results summary."""
    print("\n=== GRAM MATRIX DECODER ARS ANALYSIS ===")
    print("Condition     | Position Acc | ARS Lower Bound")
    print("-" * 50)
    
    for _, row in summary_df.iterrows():
        condition = row['condition']
        pos_acc = row['position_accuracy']
        ars = row['ARS_lb']
        
        print(f"{condition:12s} | {pos_acc:.3f}        | {ars:.3f}")


def main():
    """
    Main function to run the complete entropy reduction analysis.
    """
    print("Entropy Reduction Analysis - Ambiguity-Reduction Score (ARS)")
    print("=" * 75)
    
    # === CLASSIFICATION ANALYSIS (Standard Decoder) ===
    try:
        print("\nLoading classification results...")
        df_classid = load_classification_results(
            "data/output-classes/classid-decoding-accuracy-final.csv"
        )
        
        classid_summary = summarize_classification_results(df_classid)
        print_classification_summary(classid_summary)
        
    except FileNotFoundError as e:
        print(f"Error loading classification data: {e}")
    except Exception as e:
        print(f"Error in classification analysis: {e}")
    
    # === REGRESSION ANALYSIS ===
    try:
        print("\nLoading regression results...")
        distances = calculate_distance_from_center_targets()
        
        input_results = load_input_pixel_results(
            ['dropout', 'no-dropout', 'untrained'],
            targets=distances,
            data_dir="data/input-pixels"
        )
        
        if input_results:
            regression_summary = analyze_regression_ars(input_results)
            print_regression_summary(regression_summary)
        else:
            print("No regression data could be loaded.")
            
    except Exception as e:
        print(f"Error in regression analysis: {e}")
    
    # === GRAM MATRIX DECODER ANALYSIS ===
    try:
        print("\nRunning gram matrix decoder analysis...")
        print("(This may take a few minutes as it runs the comparison experiment)")
        
        accuracies, accuracy_stds, model_names = run_gram_matrix_comparison()
        
        if accuracies:
            gram_summary = analyze_gram_matrix_ars(accuracies, model_names)
            print_gram_matrix_summary(gram_summary)
        else:
            print("Could not generate gram matrix decoder results.")
            
    except Exception as e:
        print(f"Error in gram matrix decoder analysis: {e}")
    
    print("\n" + "=" * 75)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
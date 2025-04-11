import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils.confusion_matrix_processor import ConfusionMatrixProcessor

def main():
    """
    Test script for the ConfusionMatrixProcessor class.
    This script demonstrates how to use the class with sample data.
    """
    # Create sample data for testing
    # Ground truth dataframe
    gt_data = {
        'col1': ['A', 'B', 'C', 'A', 'B', 'A', 'C', 'D', 'E', 'A'],
        'col2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X'],
    }
    gt_df = pd.DataFrame(gt_data)
    
    # Prediction dataframe with some errors
    pred_data = {
        'col1': ['A', 'B', 'C', 'B', 'B', 'A', 'D', 'D', 'C', 'A'],  # 3 errors
        'col2': ['X', 'X', 'Z', 'X', 'Y', 'X', 'X', 'Y', 'Z', 'Z'],  # 3 errors
    }
    pred_df = pd.DataFrame(pred_data)
    
    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the processor
    processor = ConfusionMatrixProcessor(
        case_sensitive=False,
        output_dir=str(output_dir),
        verbose=True,
        enable_logging=False
    )
    
    print("=== Testing Binary Confusion Matrix ===")
    # Test create_binary_confusion_matrix for value 'A' in col1
    binary_cm = processor.create_binary_confusion_matrix(
        gt_df['col1'], 
        pred_df['col1'], 
        'A'
    )
    print(f"Binary confusion matrix for 'A':")
    print(f"True Positives: {binary_cm['true_positives']}")
    print(f"False Positives: {binary_cm['false_positives']}")
    print(f"False Negatives: {binary_cm['false_negatives']}")
    print(f"True Negatives: {binary_cm['true_negatives']}")
    print(f"Precision: {binary_cm['precision']:.2f}")
    print(f"Recall: {binary_cm['recall']:.2f}")
    print(f"F1 Score: {binary_cm['f1_score']:.2f}")
    
    print("\n=== Testing Multiclass Confusion Matrix ===")
    # Test create_multiclass_confusion_matrix
    multi_cm = processor.create_multiclass_confusion_matrix(
        gt_df, 
        pred_df, 
        ['col1', 'col2']
    )
    print("Multiclass confusion matrix:")
    print(multi_cm)
    
    print("\n=== Testing One-vs-All Analysis ===")
    # Test one_vs_all_analysis
    analysis = processor.one_vs_all_analysis(
        gt_df, 
        pred_df, 
        ['col1', 'col2'], 
        unique_id="test_case"
    )
    print("One-vs-All Analysis Summary:")
    print(analysis['summary'])
    
    print("\n=== Testing Multiple Analyses and Aggregation ===")
    # Create another test case with different data
    gt_data2 = {
        'col1': ['A', 'B', 'C', 'D', 'E'],
        'col2': ['X', 'Y', 'Z', 'X', 'Y'],
    }
    gt_df2 = pd.DataFrame(gt_data2)
    
    pred_data2 = {
        'col1': ['A', 'B', 'A', 'D', 'C'],  # 2 errors
        'col2': ['Z', 'Y', 'Z', 'X', 'Y'],  # 2 errors
    }
    pred_df2 = pd.DataFrame(pred_data2)
    
    # Run analysis on second test case
    analysis2 = processor.one_vs_all_analysis(
        gt_df2, 
        pred_df2, 
        ['col1', 'col2'], 
        unique_id="test_case2"
    )
    
    # Aggregate results from multiple analyses
    combined_results = {
        "test_case1": analysis,
        "test_case2": analysis2
    }
    
    aggregated_df = processor.aggregate_results(combined_results)
    print("Aggregated Results:")
    print(aggregated_df)
    
    print("\n=== Testing Visualization ===")
    try:
        # Create a simple confusion matrix for visualization
        cm = pd.DataFrame({
            'A': [3, 1, 0],
            'B': [1, 2, 1],
            'C': [0, 0, 2]
        }, index=['A', 'B', 'C'])
        
        processor.visualize_confusion_matrix(
            cm,
            title="Test Confusion Matrix",
            save_path=str(output_dir / "test_confusion_matrix.png")
        )
        print("Visualization saved to test_output/test_confusion_matrix.png")
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
    
    print("\nAll tests completed. Check test_output directory for results.")

if __name__ == "__main__":
    main() 
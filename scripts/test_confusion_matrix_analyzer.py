"""
Script to test the ConfusionMatrixAnalyzer by loading and analyzing dataframes from two folders.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import re

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.confusion_matrix_processing.confusion_matrix_analyzer import ConfusionMatrixAnalyzer
from utils.data_cleaning.clean_manager import CleanManager
from utils.path_manager import PathManager
from scripts.define_value_replacements import get_value_replacements, get_numeric_columns

# Define columns to process (same as in ConfusionMatrixSessionProcessor)
COLS_TO_PROCESS = ['L1', 'L2', 'L3', 'L4', 'L5', 
                  'L6', 'L7', 'L8', 'L9', 'L10',
                  'L11', 'L12', 'L13', 'L14', 'L15',
                  'L16', 'L17', 'L18', 'L19', 'L20']

def clean_dataframe(df: pd.DataFrame, replace_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Clean a DataFrame by replacing specified values.
    
    Args:
        df: DataFrame to clean
        replace_dict: Dictionary of values to replace
        
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        if col in COLS_TO_PROCESS:
            cleaned_df[col] = cleaned_df[col].replace(replace_dict)
    return cleaned_df

def extract_identifier(filename: str) -> str:
    """
    Extract the row/column identifier from a filename.
    Example: 'arget_singer_gt_20240814_R10P14_R10P22.csv' -> 'R10P14_R10P22'
    """
    # Match pattern R#P#_R#P# where # is any number
    match = re.search(r'R\d+P\d+_R\d+P\d+', filename)
    if match:
        return match.group(0)
    print(f"Warning: Could not find R#P#_R#P# pattern in {filename}")
    return None

def load_dataframes_from_folder(folder_path: Path, replace_dict: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a folder into DataFrames.
    
    Args:
        folder_path: Path to the folder containing CSV files
        replace_dict: Dictionary of values to replace (optional)
        
    Returns:
        Dictionary mapping identifiers to DataFrames
    """
    dataframes = {}
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return dataframes
        
    for file_path in folder_path.glob("*.csv"):
        try:
            identifier = extract_identifier(file_path.stem)
            if identifier:
                df = pd.read_csv(file_path)
                if replace_dict:
                    df = clean_dataframe(df, replace_dict)
                dataframes[identifier] = df
                print(f"Loaded {file_path.name} with shape {df.shape} (identifier: {identifier})")
            else:
                print(f"Warning: Could not extract identifier from {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
            
    return dataframes

def filter_columns(df: pd.DataFrame) -> List[str]:
    """
    Filter columns to process based on COLS_TO_PROCESS and available columns.
    
    Args:
        df: DataFrame to filter columns from
        
    Returns:
        List of columns that exist in both COLS_TO_PROCESS and the DataFrame
    """
    return [col for col in COLS_TO_PROCESS if col in df.columns]

def main():
    # Set up output directory for analysis results
    output_dir = project_root / "analysis_results" / "confusion_matrix_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ConfusionMatrixAnalyzer(
        output_dir=output_dir,
        case_sensitive=False  # Set to True if you want case-sensitive comparisons
    )
    
    # Get value replacements
    replace_dict = get_value_replacements()
    
    # Get input folder paths
    gt_folder = Path("arget_singer/20250421/20250421_161923/ground_truth")
    pred_folder = Path("arget_singer/20250421/20250421_161923/processed/checkpoints/ckpt5_cleaned")
    
    # Load ground truth dataframes with cleaning
    print("\nLoading and cleaning ground truth dataframes...")
    ground_truth_dfs = load_dataframes_from_folder(gt_folder, replace_dict)
    
    if not ground_truth_dfs:
        print("No ground truth dataframes found to process")
        return
        
    # Load prediction dataframes
    print("\nLoading prediction dataframes...")
    prediction_dfs = load_dataframes_from_folder(pred_folder)
    
    if not prediction_dfs:
        print("No prediction dataframes found to process")
        return
        
    # Find common identifiers
    common_identifiers = set(ground_truth_dfs.keys()) & set(prediction_dfs.keys())
    if not common_identifiers:
        print("Error: No matching identifiers found between ground truth and prediction files")
        return
        
    print(f"\nFound {len(common_identifiers)} matching pairs of files")
    
    # Create filtered dictionaries with only matching pairs
    matched_ground_truth = {id: ground_truth_dfs[id] for id in common_identifiers}
    matched_predictions = {id: prediction_dfs[id] for id in common_identifiers}
    
    # Process each file pair
    for identifier in common_identifiers:
        gt_df = matched_ground_truth[identifier]
        pred_df = matched_predictions[identifier]
        
        # Get columns to analyze for this pair
        columns_to_analyze = filter_columns(gt_df)
        columns_to_analyze = [col for col in columns_to_analyze if col in pred_df.columns]
        
        if not columns_to_analyze:
            print(f"Warning: No matching columns found in both DataFrames for {identifier}")
            continue
            
        print(f"\nAnalyzing {identifier} with columns: {columns_to_analyze}")
        
        # Run analysis for this file pair
        results = analyzer.analyze_dataframes(
            gt_df=gt_df,
            pred_df=pred_df,
            columns=columns_to_analyze,
            identifier=identifier,
            save_results=True,
            create_visualizations=True,
            aggregate_results=True
        )
        
        if results.get('file_paths', {}).get('confusion_matrix'):
            print(f"Matrix saved for {identifier}: {results['file_paths']['confusion_matrix']}")
    
    # Finalize analysis with aggregation
    print("\nFinalizing analysis with aggregation...")
    final_results = analyzer.finalize_analysis(identifier="dataset_overview")
    
    if final_results["status"] == "success":
        print("\nAnalysis completed successfully!")
        print(f"Total matrices combined: {final_results['matrices_combined']}")
        print(f"Total predictions analyzed: {final_results['total_predictions']}")
        print("\nGenerated files:")
        for file_type, path in final_results["file_paths"].items():
            print(f"  {file_type}: {path}")
    else:
        print(f"\nError during analysis: {final_results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main() 
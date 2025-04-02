import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.azure_config import AZURE_FORM_RECOGNIZER_ENDPOINT, AZURE_FORM_RECOGNIZER_KEY
from utils.ocr_processor import OCRProcessor
from utils.comparator import DataColumnComparator, IndexColumnComparator
from collections import defaultdict
from utils.data_preprocessors import DataLoader, DimensionComparison
from utils.col_idx_processing import ArgetSinger24ColIdxProcessor
# Load environment variables
load_dotenv()

# Get the project root directory
root_dir = Path(__file__).parent
data_dir = root_dir / "data"
output_dir = root_dir / "output"
output_dir.mkdir(exist_ok=True)

# Define paths
BASE_DIR = root_dir
DATA_DIR = data_dir
GT_DIR = DATA_DIR / "ground_truth"

def setup_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    GT_DIR.mkdir(exist_ok=True)

def validate_csv_format(gt_path: str) -> None:
    """Validate the format of the ground truth CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(gt_path)
        
        # Basic validation - can be expanded later
        required_columns = ['date', 'row', 'panel', 'disease'] + [f'L{i}' for i in range(1, 21)]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ground truth CSV: {missing_cols}")
            
    except pd.errors.EmptyDataError:
        raise ValueError("Ground truth CSV file is empty")
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format in ground truth file")

def validate_paths(image_path: str, gt_path: Path) -> None:
    """Validate that input files exist."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not str(gt_path).endswith('.csv'):
        raise ValueError("Ground truth file must be a CSV file")
    
    # Validate CSV format
    validate_csv_format(str(gt_path))

def print_confusion_matrix(confusion_matrix: pd.DataFrame, verbose: bool = True) -> None:
    """Print the confusion matrix in a formatted way with detailed analysis."""
    if not verbose:
        return
        
    print("\n=== Confusion Matrix Analysis ===")
    print("Rows: Ground Truth Values")
    print("Columns: OCR Values")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    # Calculate summary statistics
    total = confusion_matrix.sum().sum()
    correct = sum(confusion_matrix[i][i] for i in confusion_matrix.index if i in confusion_matrix.columns)
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("\n=== Summary Statistics ===")
    print(f"Total Cells: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Analyze most common errors
    print("\n=== Error Analysis ===")
    error_counts = defaultdict(int)
    for gt_val in confusion_matrix.index:
        for ocr_val in confusion_matrix.columns:
            if gt_val != ocr_val:
                count = confusion_matrix.loc[gt_val, ocr_val]
                if count > 0:
                    error_counts[(gt_val, ocr_val)] = count
    
    if error_counts:
        print("\nMost Common Errors (Ground Truth → OCR):")
        for (gt_val, ocr_val), count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {gt_val} → {ocr_val}: {count} occurrences")
    
    # Analyze per-value accuracy
    print("\n=== Per-Value Accuracy ===")
    for val in confusion_matrix.index:
        if val in confusion_matrix.columns:
            total_val = confusion_matrix.loc[val].sum()
            correct_val = confusion_matrix.loc[val, val]
            if total_val > 0:
                accuracy_val = (correct_val / total_val) * 100
                print(f"Value '{val}': {accuracy_val:.2f}% accuracy ({correct_val}/{total_val} correct)")
    
    # Calculate precision and recall for each value
    print("\n=== Precision and Recall ===")
    for val in confusion_matrix.index:
        if val in confusion_matrix.columns:
            # True Positives
            tp = confusion_matrix.loc[val, val]
            # False Positives (sum of column minus true positives)
            fp = confusion_matrix[val].sum() - tp
            # False Negatives (sum of row minus true positives)
            fn = confusion_matrix.loc[val].sum() - tp
            
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            
            print(f"\nValue '{val}':")
            print(f"  Precision: {precision:.2f}%")
            print(f"  Recall: {recall:.2f}%")
            print(f"  F1 Score: {2 * (precision * recall) / (precision + recall):.2f}%")

def print_processing_summary(comparison_results: dict, verbose: bool = True) -> None:
    """Print the processing summary and mismatches."""
    if not verbose:
        return
        
    print("\n=== Processing Summary ===")
    print(f"Accuracy: {comparison_results['value_comparison']['accuracy']:.2f}%")
    print(f"Total Cells Compared: {comparison_results['value_comparison']['total_cells']}")
    print(f"Matching Cells: {comparison_results['value_comparison']['matches']}")

def print_index_analysis(index_analysis: Dict[str, Any], ocr_df: pd.DataFrame, verbose: bool = False) -> None:
    """Print analysis of index columns.
    
    Args:
        index_analysis: Dictionary containing index column analysis results
        ocr_df: DataFrame containing OCR results
        verbose: Whether to print detailed output
    """
    if not verbose:
        return
        
    print("\n" + "="*50)
    print("INDEX COLUMN ANALYSIS")
    print("="*50)
    
    # Print OCR DataFrame columns
    print("\nOCR DataFrame Columns:")
    print("-"*30)
    for idx, col in enumerate(ocr_df.columns):
        print(f"Index {idx}: {col}")
    print("-"*30)
    
    # Print presence check results
    print("\nIndex Column Comparison:")
    print("-"*30)
    presence_check = index_analysis.get('presence_check', {})
    for col, info in presence_check.items():
        print(f"\n{col} (Index {info['index']}):")
        print(f"  OCR Column: {info['ocr_col_name']}")
        print(f"  GT Column:  {info['gt_col_name']}")
        
        if info['ocr_values']:
            print("  OCR Values:")
            for idx, val in enumerate(info['ocr_values'][:5]):  # Show first 5 values
                print(f"    {idx+1}: {val}")
            if len(info['ocr_values']) > 5:
                print(f"    ... and {len(info['ocr_values']) - 5} more")
                
        if info['gt_values']:
            print("  GT Values:")
            for idx, val in enumerate(info['gt_values'][:5]):  # Show first 5 values
                print(f"    {idx+1}: {val}")
            if len(info['gt_values']) > 5:
                print(f"    ... and {len(info['gt_values']) - 5} more")
    
    # Print validation results
    print("\nValidation Results:")
    print("-"*30)
    validation_results = index_analysis.get('validation_results', {})
    for col, analysis in validation_results.items():
        if isinstance(analysis, dict) and 'error' in analysis:
            print(f"  ❌ {col} (Index {analysis['index']}): {analysis['error']}")
        else:
            print(f"  ✅ {col} (Index {analysis['index']}):")
            print(f"    Total Entries: {analysis['total_entries']}")
            print(f"    Correct Entries: {analysis['correct_entries']}")
            print(f"    Accuracy: {analysis['accuracy']:.2f}%")
            
            # Print error details
            if 'format_errors' in analysis and analysis['format_errors']:
                print(f"    Format Errors: {len(analysis['format_errors'])}")
                for error in analysis['format_errors'][:5]:
                    print(f"      - {error['error']}")
                if len(analysis['format_errors']) > 5:
                    print(f"      ... and {len(analysis['format_errors']) - 5} more")
                    
            if 'range_errors' in analysis and analysis['range_errors']:
                print(f"    Range Errors: {len(analysis['range_errors'])}")
                for error in analysis['range_errors'][:5]:
                    print(f"      - {error['error']}")
                if len(analysis['range_errors']) > 5:
                    print(f"      ... and {len(analysis['range_errors']) - 5} more")
                    
            if 'category_errors' in analysis and analysis['category_errors']:
                print(f"    Category Errors: {len(analysis['category_errors'])}")
                for error in analysis['category_errors'][:5]:
                    print(f"      - {error['error']}")
                if len(analysis['category_errors']) > 5:
                    print(f"      ... and {len(analysis['category_errors']) - 5} more")
                    
            if 'spelling_errors' in analysis and analysis['spelling_errors']:
                print(f"    Spelling Errors: {len(analysis['spelling_errors'])}")
                for error in analysis['spelling_errors'][:5]:
                    print(f"      - {error['error']}")
                if len(analysis['spelling_errors']) > 5:
                    print(f"      ... and {len(analysis['spelling_errors']) - 5} more")
                    
            if 'sequence_errors' in analysis and analysis['sequence_errors']:
                print(f"    Sequence Errors: {len(analysis['sequence_errors'])}")
                for error in analysis['sequence_errors'][:5]:
                    print(f"      - {error['error']}")
                if len(analysis['sequence_errors']) > 5:
                    print(f"      ... and {len(analysis['sequence_errors']) - 5} more")
def display_validation_results(is_valid: bool, message: str, col_info: dict, gt_df: pd.DataFrame, 
                                     ocr_df: pd.DataFrame, data_processor) -> None:
        """Display validation results and DataFrame information.
        
        Args:
            is_valid: Whether validation passed
            message: Validation message
            col_info: Column validation information
            gt_df: Ground truth DataFrame
            ocr_df: OCR DataFrame
            data_processor: Data processor instance with column definitions
        """
        if not is_valid:
            print(f"\nValidation failed: {message}\n Column Information: {col_info}")
            return
        else:
            print(f"\nValidation successful: {message}")
            
            # Display DataFrame information
            print("\nGround Truth DataFrame:")
            print(gt_df.head())
            print("\nOCR DataFrame:")
            print(ocr_df.head())
            
            # Display column information
            print("\nColumn Information:")
            print("Index Columns:", data_processor.index_cols)
            print("Data Columns:", data_processor.data_cols)
def main():
    """Main entry point for the OCR pipeline."""
    # Initialize processors
    data_loader = DataLoader(verbose=False)
    column_processor = DimensionComparison(verbose=True)
    ocr_processor = OCRProcessor()
    
    # Define the paths
    gt_path = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ground_truth/ground_truth.csv")
    ocr_path = "/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ocr_predictions/arget_singer_24_ocr_pred.csv"
    
    # Uncomment and modify path to process a new image with OCR
    image_path = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/arget_singer_24.png")
    # try:
    #     tables = ocr_processor.process_document(image_path)
    #     print(f"Successfully processed {len(tables)} tables from image")
    #     
    # except Exception as e:
    #     print(f"Error processing image with OCR: {str(e)}")
    #     
    
    try:
        # Check if files exist
        if not gt_path.exists():
            print(f"Error: Ground truth file not found at {gt_path}")
            return
            
        if not os.path.exists(ocr_path):
            print(f"Error: OCR file not found at {ocr_path}")
            return
            
        # Load DataFrames
        gt_df = data_loader.load_df(gt_path)
        ocr_df = data_loader.load_df(ocr_path)
        
        # Reset index for OCR DataFrame
        ocr_df = data_loader.reset_index(ocr_df)
        
        # Compare dimensions with ground truth
        is_valid, message = column_processor.compare_dimensions(gt_df, ocr_df)
        if not is_valid:
            print(f"\nWarning: {message}")
            return
        else:
            print(f"\nDimension check passed: {message}\n Starting Data Processing...")
        
        # Initialize data processors
        data_processor = ArgetSinger24ColIdxProcessor(verbose=True)
        data_processor.set_dataframes(gt_df, ocr_df)
        
        # Process data
        is_valid, message, col_info = data_processor.validate_dataframes()

        display_validation_results(is_valid, message, col_info, gt_df, ocr_df, data_processor)

        if not is_valid:
            ocr_df,_ = data_processor.rename_columns_by_index()
            is_valid, message, col_info = data_processor.validate_dataframes()
            display_validation_results(is_valid, message, col_info, gt_df, ocr_df, data_processor)
            data_processor.save_to_csv(ocr_df, ocr_path, prefix="col_matched_")
    except FileNotFoundError as e:
        print(f"Error: File not found: {str(e)}")
    except ValueError as e:
        print(f"Error: Invalid file format: {str(e)}")
    except Exception as e:
        print(f"Error processing data: {str(e)}")


if __name__ == "__main__":
    main() 
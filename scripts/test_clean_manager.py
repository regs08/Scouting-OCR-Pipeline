"""
Script to test the CleanManager by loading and cleaning dataframes from a folder.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_cleaning.clean_manager import CleanManager
from utils.path_manager import PathManager
from scripts.define_value_replacements import get_value_replacements, get_numeric_columns

def load_dataframes_from_folder(folder_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a folder into DataFrames.
    
    Args:
        folder_path: Path to the folder containing CSV files
        
    Returns:
        Dictionary mapping filenames to DataFrames
    """
    dataframes = {}
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return dataframes
        
    for file_path in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(file_path)
            dataframes[file_path.stem] = df
            print(f"Loaded {file_path.name} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
            
    return dataframes

def setup_test_directory(path_manager: PathManager, session_id: str) -> None:
    """
    Set up the test directory structure.
    
    Args:
        path_manager: PathManager instance
        session_id: Session ID for the test
    """
    # Get session paths
    session_paths = path_manager.get_session_paths(session_id)
    
    # Create necessary directories
    directories = [
        session_paths['raw'],
        session_paths['processed'],
        session_paths['flagged'],
        session_paths['logs']
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    # Initialize path manager with test parameters
    path_manager = PathManager(
        vineyard="arget_singer",
        batch="20250421"
    )
    
    # Get test session ID
    session_id = "20250421_161923"
    
    # Set up test directory structure
    print("\nSetting up test directory structure...")
    setup_test_directory(path_manager, session_id)
    
    # Initialize clean manager
    clean_manager = CleanManager(
        path_manager=path_manager,
        session_id=session_id,
        verbose=True
    )
    
    # Get predefined replacements and columns
    replace_dict = get_value_replacements()
    numeric_cols = get_numeric_columns()
    
    # Get input folder path
    input_folder = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/input/ground_truth")
    
    # Load dataframes
    print("\nLoading dataframes...")
    dataframes = load_dataframes_from_folder(input_folder)
    
    if not dataframes:
        print("No dataframes found to process")
        return
        
    # Prepare input data
    input_data = {
        "dataframes": dataframes,
        "numeric_cols": numeric_cols,
        "replace_dict": replace_dict
    }
    
    # Run cleaning process
    print("\nRunning cleaning process...")
    result = clean_manager.process(input_data)
    
    # Check for errors
    if "error" in result:
        print(f"Error during cleaning: {result['error']}")
        return
        
    # Print cleaning statistics
    print("\nCleaning Statistics:")
    for filename, stats in result["cleaning_stats"].items():
        print(f"\n{filename}:")
        print(f"  Original shape: {stats['original_shape']}")
        print(f"  Cleaned shape: {stats['cleaned_shape']}")
        print(f"  Changes made: {stats['changes_made']}")
        
    # Print sample of cleaned data
    print("\nSample of cleaned data:")
    for filename, df in result["cleaned_dataframes"].items():
        print(f"\n{filename}:")
        print(df.head())
        
    print("\nCleaning process completed successfully!")

if __name__ == "__main__":
    main() 
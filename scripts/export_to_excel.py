"""
Script to export CSV files to Excel format, flagging all non-zero values.
"""

import sys
from pathlib import Path
import logging
import os

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.excel_exporter import ExcelExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def process_folder(folder_path: Path, exporter: ExcelExporter) -> None:
    """
    Process all CSV files in a folder and convert them to Excel format.
    
    Args:
        folder_path: Path to the folder containing CSV files
        exporter: ExcelExporter instance to use
    """
    try:
        # Resolve the path to handle any symlinks and get absolute path
        folder_path = folder_path.resolve()
        
        if not folder_path.exists():
            print(f"Error: Folder not found: {folder_path}")
            return
            
        print(f"\nSearching for CSV files in: {folder_path}")
        
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return
            
        print(f"\nFound {len(csv_files)} CSV files to process:")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
        
        # Process each CSV file
        for csv_path in csv_files:
            try:
                print(f"\nProcessing: {csv_path.name}")
                output_path = exporter.export_csv_to_excel(
                    csv_path,
                    sheet_name="Confusion Matrix",
                    index=False
                )
                print(f"Exported to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {csv_path.name}: {str(e)}")
                
    except Exception as e:
        print(f"Error accessing folder: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_to_excel.py <path_to_folder>")
        sys.exit(1)
    
    # Get folder path and handle spaces correctly
    raw_path = " ".join(sys.argv[1:])  # Join all arguments to handle paths with spaces
    folder_path = Path(os.path.expanduser(raw_path))  # Expand user directory if needed
    
    try:
        # Initialize exporter with output in the same folder as input
        exporter = ExcelExporter(output_dir=folder_path)
        
        # Process all CSV files in the folder
        process_folder(folder_path, exporter)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
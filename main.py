import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import re
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.azure_config import AZURE_FORM_RECOGNIZER_ENDPOINT, AZURE_FORM_RECOGNIZER_KEY
from utils.ocr_processor import OCRProcessor
from collections import defaultdict, namedtuple
from utils.data_preprocessors import DataLoader, DimensionComparison
from utils.col_idx_processing import ArgetSinger24ColIdxProcessor
from utils.file_search.file_matcher import FileMatcher
from utils.base_processor import BaseProcessor
from utils.confusion_matrix_processor import ConfusionMatrixProcessor
from utils.data_cleaning.basic_cleaner import BasicCleaner
from utils.data_cleaning.simple_value_cleaner import SimpleValueCleaner
from utils.path_manager import PathManager
from utils.setup_manager import SetupManager
import sys
from utils.session_manager import SessionManager
from utils.directory_manager import DirectoryManager

# Load environment variables
load_dotenv()

# Define base directories
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
SYSTEM_LOGS_DIR = PROJECT_ROOT / "system_logs"

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Main entry point for the OCR pipeline."""
    # Initialize managers
    setup_manager = SetupManager(
        input_dir=Path("input"),
        expected_vineyard="arget_singer",
        verbose=True
    )
    
    # Set up session and get directory info
    session_dir, session_logs_dir, data_files = setup_manager.setup_session()
    if not session_dir:
        print("Failed to set up session")
        return
        
    # Get vineyard and date from session directory
    vineyard = session_dir.parent.parent.name  # Get vineyard from grandparent directory
    date = session_dir.parent.name  # Get date from parent directory
    session_id = session_dir.name  # Get session ID from directory name
    
    # Initialize path manager with vineyard and date
    path_manager = PathManager(vineyard=vineyard, batch=date)
    
    # Initialize directory manager
    directory_manager = DirectoryManager(
        verbose=True,
        enable_logging=True,
        enable_console=True,
        log_dir=session_logs_dir,
        operation_name="main_directory_manager"
    )
    
    # Initialize session manager with the correct session directory
    session_manager = SessionManager(
        path_manager=path_manager,
        session_id=session_id,  # Use the session ID from the directory
        verbose=True
    )
    
    # Initialize OCR processor
    ocr_processor = OCRProcessor(
        path_manager=path_manager,
        session_id=session_id,  # Pass the session ID to the processor
        verbose=True
    )
    
    # Initialize dimension comparison processor
    dim_processor = DimensionComparison(
        path_manager=path_manager,
        session_id=session_id,
        verbose=True
    )
    
    # Add processors to pipeline
    session_manager.add_processor(ocr_processor, "ckpt1_ocr_processed", 1)
    session_manager.add_processor(dim_processor, "ckpt2_dimension_check", 2)
    
    # # Run the pipeline
    
    session_manager.process_session()

if __name__ == "__main__":
    main() 
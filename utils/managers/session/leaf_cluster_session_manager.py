"""
Session manager for leaf cluster OCR processing.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd

from utils.managers.session.base_session_manager import BaseSessionManager
from utils.path_manager import PathManager
from utils.session_processors.ocr_processor import OCRProcessor

class LeafClusterSessionManager(BaseSessionManager):
    """
    Manager for processing leaf cluster OCR in a session.
    This is a simplified manager that only handles OCR processing.
    """
    
    def __init__(self,
                 path_manager: PathManager,
                 session_id: str,
                 expected_data_cols: List[str],
                 expected_index_cols: List[str],
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the leaf cluster session manager.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            expected_data_cols: List of expected data column names
            expected_index_cols: List of expected index column names
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            expected_data_cols=expected_data_cols,
            expected_index_cols=expected_index_cols,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="leaf_cluster_ocr"
        )
        
    def _setup_pipeline(self) -> None:
        """Set up the OCR processing pipeline."""
        # Create OCR processor
        ocr_processor = OCRProcessor(
            path_manager=self.path_manager,
            session_id=self.session_id,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir
        )
        
        # Add OCR processor as the only component
        self.add_component(
            component=ocr_processor,
            checkpoint_name="ckpt1_ocr",
            checkpoint_number=1
        )
        
    def _load_checkpoint_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from a checkpoint file.
        
        Args:
            file_path: Path to the checkpoint file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(file_path)
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process leaf cluster images through OCR.
        
        Args:
            input_data: Dictionary containing:
                - session_dir: Path to session directory
                - image_paths: List of paths to images to process
                
        Returns:
            Dictionary containing OCR results and processing information
        """
        self.log_info("process", f"Starting leaf cluster OCR processing for session {self.session_id}")
        
        try:
            # Validate input data
            if 'session_dir' not in input_data:
                raise ValueError("Missing session_dir in input data")
            if 'image_paths' not in input_data:
                raise ValueError("Missing image_paths in input data")
                
            # Run the pipeline (which only contains OCR processing)
            result = self.run_pipeline(input_data)
            
            self.log_info("process", "Leaf cluster OCR processing completed successfully")
            return result
            
        except Exception as e:
            self.log_error("process", f"Error during leaf cluster OCR processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the leaf cluster OCR processing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        return self.process(input_data) 
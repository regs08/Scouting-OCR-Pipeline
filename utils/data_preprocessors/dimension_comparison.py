from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd
import shutil
from utils.data_preprocessors.base_data_processor import BaseDataProcessor
import re

class DimensionComparison(BaseDataProcessor):
    """Processor for comparing dimensions between ground truth and OCR DataFrames."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the dimension comparison processor.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "dimension_comparison"
        )
        
        # Create checkpoint directory for dimension comparison
        self.checkpoint_dir = self.path_manager.get_checkpoint_path(
            session_id=self.session_id,
            checkpoint="ckpt2_dimension_comparison"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create flagged directory for mismatched files
        self.flagged_dir = self.path_manager.get_flagged_path(
            session_id=self.session_id,
            reason="dimension_mismatch"
        )
        self.flagged_dir.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def get_input_type(cls) -> type:
        """Get the expected input type for this processor."""
        return dict
        
    @classmethod
    def get_output_type(cls) -> type:
        """Get the output type for this processor."""
        return dict
        
    def _load_ground_truth(self, session_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load ground truth data from the session directory.
        
        Args:
            session_dir: Path to the session directory
            
        Returns:
            Dictionary mapping filenames to ground truth DataFrames
        """
        gt_dir = session_dir / "ground_truth"
        if not gt_dir.exists():
            self.log_warning("_load_ground_truth", "No ground truth directory found")
            return {}
            
        gt_data = {}
        for gt_file in gt_dir.glob("*.csv"):
            try:
                df = pd.read_csv(gt_file)
                gt_data[gt_file.stem] = df
                self.log_info("_load_ground_truth", f"Loaded ground truth data from {gt_file.name}")
            except Exception as e:
                self.log_error("_load_ground_truth", f"Error loading {gt_file.name}: {str(e)}")
                
        return gt_data
        
    def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Tuple[bool, str]]:
        """
        Compare dimensions between ground truth and OCR DataFrames.
        
        Args:
            input_data: Dictionary containing session data and previous checkpoint data
            
        Returns:
            Dictionary mapping file names to (is_valid, message) tuples
        """
        results = {}
        
        # Get session directory from path manager
        session_dir = self.path_manager.base_dir / self.session_id
        
        # Get OCR data from previous checkpoint and remove first row from each DataFrame
        ocr_data = input_data.get('prev_checkpoint_data', {})
        if not ocr_data:
            self.log_error("_process_impl", "No OCR data found from previous checkpoint")
            return {}
            
        # Remove first row from all OCR DataFrames
        for filename, df in ocr_data.items():
            ocr_data[filename] = df.iloc[1:].reset_index(drop=True)
            self.log_info("_process_impl", f"Removed first row from {filename}")
        
        # Load ground truth data
        gt_data = self._load_ground_truth(session_dir)
        
        if not gt_data:
            self.log_error("_process_impl", "No ground truth data found")
            return {filename: (False, "No ground truth data available") for filename in ocr_data.keys()}
        
        # Get the source directory for OCR files
        source_dir = self.path_manager.get_checkpoint_path(self.session_id, "ckpt1_ocr_processed")
        
        # Create a mapping of sequence patterns to ground truth files
        gt_sequence_map = {}
        for gt_filename, gt_df in gt_data.items():
            # Extract sequence pattern (R#P#_R#P#) from ground truth filename
            sequence_match = re.search(r'R\d+P\d+_R\d+P\d+', gt_filename)
            if sequence_match:
                sequence = sequence_match.group()
                gt_sequence_map[sequence] = gt_df
                self.log_info("_process_impl", f"Mapped sequence {sequence} to ground truth file {gt_filename}")
        
        for filename, ocr_df in ocr_data.items():
            try:
                # Extract sequence pattern (R#P#_R#P#) from OCR filename
                sequence_match = re.search(r'R\d+P\d+_R\d+P\d+', filename)
                if not sequence_match:
                    results[filename] = (False, "No sequence pattern found in filename")
                    continue
                    
                sequence = sequence_match.group()
                gt_df = gt_sequence_map.get(sequence)
                
                if gt_df is None:
                    results[filename] = (False, f"No ground truth data found for sequence {sequence}")
                    # Copy to flagged directory
                    source_file = source_dir / f"{filename}.csv"
                    if source_file.exists():
                        target_file = self.flagged_dir / f"{filename}.csv"
                        ocr_df.to_csv(target_file, index=False)
                        self.log_warning("_process_impl", f"Copied {filename} to flagged directory: No ground truth data for sequence {sequence}")
                    continue
                    
                # Compare dimensions
                gt_rows, gt_cols = gt_df.shape
                ocr_rows, ocr_cols = ocr_df.shape
                
                if self.verbose:
                    self.log_info("compare_dimensions", 
                                f"\n=== Dimension Comparison for {filename} ===")
                    self.log_info("compare_dimensions", 
                                f"Ground Truth: {gt_rows} rows × {gt_cols} columns")
                    self.log_info("compare_dimensions", 
                                f"OCR Data: {ocr_rows} rows × {ocr_cols} columns")
                
                # Check if column counts match
                if gt_cols != ocr_cols:
                    results[filename] = (False, f"Column count mismatch: GT={gt_cols}, OCR={ocr_cols}")
                    # Copy to flagged directory
                    source_file = source_dir / f"{filename}.csv"
                    if source_file.exists():
                        target_file = self.flagged_dir / f"{filename}.csv"
                        ocr_df.to_csv(target_file, index=False)
                        self.log_warning("_process_impl", f"Copied {filename} to flagged directory: Column count mismatch")
                    continue
                
                # Check if row counts match
                if gt_rows != ocr_rows:
                    results[filename] = (False, f"Row count mismatch: GT={gt_rows}, OCR={ocr_rows}")
                    # Copy to flagged directory
                    source_file = source_dir / f"{filename}.csv"
                    if source_file.exists():
                        target_file = self.flagged_dir / f"{filename}.csv"
                        ocr_df.to_csv(target_file, index=False)
                        self.log_warning("_process_impl", f"Copied {filename} to flagged directory: Row count mismatch")
                    continue
                
                # If dimensions match, move to checkpoint directory
                results[filename] = (True, "Dimensions match")
                source_file = source_dir / f"{filename}.csv"
                if source_file.exists():
                    target_file = self.checkpoint_dir / f"{filename}.csv"
                    ocr_df.to_csv(target_file, index=False)
                    source_file.unlink()  # Remove the original file
                    self.log_info("_process_impl", f"Moved {filename} to dimension comparison directory: Dimensions match")
                
            except Exception as e:
                results[filename] = (False, f"Error comparing dimensions: {str(e)}")
                # Copy to flagged directory on error
                source_file = source_dir / f"{filename}.csv"
                if source_file.exists():
                    target_file = self.flagged_dir / f"{filename}.csv"
                    ocr_df.to_csv(target_file, index=False)
                    self.log_error("_process_impl", f"Copied {filename} to flagged directory: Error during comparison")
                
        return results

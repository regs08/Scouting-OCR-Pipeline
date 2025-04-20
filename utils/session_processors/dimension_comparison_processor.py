from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd
import re
from .base_comparison_processor import BaseComparisonProcessor

class DimensionComparisonProcessor(BaseComparisonProcessor):
    """Processor for comparing dimensions between ground truth and OCR DataFrames."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
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
            case_sensitive=False, # Default for dimension comparison
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
        
    def _load_ocr_data(self, ocr_output_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load OCR data from previous checkpoint.
        
        Args:
            ocr_output_dir: Path to the OCR output directory
            
        Returns:
            Dictionary mapping filenames to OCR DataFrames
        """
        ocr_data = {}
        for ocr_file in ocr_output_dir.glob("*.csv"):
            try:
                df = pd.read_csv(ocr_file)
                # Remove first row - typically contains headers we don't need
                df = df.iloc[1:].reset_index(drop=True)
                ocr_data[ocr_file.stem] = df
                self.log_info("_load_ocr_data", f"Loaded OCR data from {ocr_file.name}")
            except Exception as e:
                self.log_error("_load_ocr_data", f"Error loading {ocr_file.name}: {str(e)}")
                
        return ocr_data
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare dimensions between ground truth and OCR DataFrames.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Updated dictionary with dimension comparison results
        """
        # Get session directory
        session_dir = input_data.get('session_dir')
        if not session_dir:
            raise ValueError("Missing session_dir in input data")
            
        session_dir = Path(session_dir)
        
        # Get OCR output directory from previous checkpoint
        ocr_output_dir = input_data.get('ocr_output_dir')
        if not ocr_output_dir:
            self.log_error("process", "No OCR output directory found in input data")
            return {"dimension_comparison_status": "error"}
            
        ocr_output_dir = Path(ocr_output_dir)
        
        # Load ground truth and OCR data
        gt_data = self._load_ground_truth(session_dir)
        ocr_data = self._load_ocr_data(ocr_output_dir)
        
        if not gt_data:
            self.log_error("process", "No ground truth data found")
            # Move all OCR files to flagged directory
            for filename, df in ocr_data.items():
                target_file = self.flagged_dir / f"{filename}.csv"
                df.to_csv(target_file, index=False)
                self.log_warning("process", f"Saved {filename} to flagged directory: No ground truth data available")
            
            return {
                "dimension_comparison_status": "completed",
                "matched_files": [],
                "mismatched_files": list(ocr_data.keys()),
                "total_files": len(ocr_data),
                "matched_count": 0,
                "mismatched_count": len(ocr_data),
                "dimension_comparison_results": {},
                "dimension_comparison_dir": str(self.checkpoint_dir)
            }
        
        if not ocr_data:
            self.log_error("process", "No OCR data found")
            return {"dimension_comparison_status": "error", "error": "No OCR data found"}
        
        # Use base class method to find matching files
        matching_files = self._find_matching_files(gt_data, ocr_data)
        
        # Handle files that exist in OCR data but not in matching_files
        unmatched_ocr_files = set(ocr_data.keys()) - set(matching_files.keys())
        for filename in unmatched_ocr_files:
            target_file = self.flagged_dir / f"{filename}.csv"
            ocr_data[filename].to_csv(target_file, index=True)
            self.log_warning("process", f"Saved {filename} to flagged directory: No matching ground truth file")
        
        if not matching_files:
            self.log_error("process", "No matching files found between ground truth and OCR data")
            return {
                "dimension_comparison_status": "completed",
                "matched_files": [],
                "mismatched_files": list(unmatched_ocr_files),
                "total_files": len(ocr_data),
                "matched_count": 0,
                "mismatched_count": len(unmatched_ocr_files),
                "dimension_comparison_results": {},
                "dimension_comparison_dir": str(self.checkpoint_dir)
            }
        
        # Compare dimensions and process files
        results = {}
        matched_files = []
        mismatched_files = []
        
        for filename, data_pair in matching_files.items():
            try:
                gt_df = data_pair['gt']
                ocr_df = data_pair['pred']
                
                # Compare dimensions
                gt_rows, gt_cols = gt_df.shape
                ocr_rows, ocr_cols = ocr_df.shape
                
                if self.verbose:
                    self.log_info("process", 
                                f"\n=== Dimension Comparison for {filename} ===")
                    self.log_info("process", 
                                f"Ground Truth: {gt_rows} rows × {gt_cols} columns")
                    self.log_info("process", 
                                f"OCR Data: {ocr_rows} rows × {ocr_cols} columns")
                
                # Check if column counts match
                if gt_cols != ocr_cols:
                    results[filename] = (False, f"Column count mismatch: GT={gt_cols}, OCR={ocr_cols}")
                    # Save to flagged directory
                    target_file = self.flagged_dir / f"{filename}.csv"
                    ocr_df.to_csv(target_file, index=False)
                    self.log_warning("process", f"Saved {filename} to flagged directory: Column count mismatch")
                    mismatched_files.append(filename)
                    continue
                
                # Check if row counts match
                if gt_rows != ocr_rows:
                    results[filename] = (False, f"Row count mismatch: GT={gt_rows}, OCR={ocr_rows}")
                    # Save to flagged directory
                    target_file = self.flagged_dir / f"{filename}.csv"
                    ocr_df.to_csv(target_file, index=False)
                    self.log_warning("process", f"Saved {filename} to flagged directory: Row count mismatch")
                    mismatched_files.append(filename)
                    continue
                
                # If dimensions match, save to checkpoint directory
                results[filename] = (True, "Dimensions match")
                target_file = self.checkpoint_dir / f"{filename}.csv"
                ocr_df.to_csv(target_file, index=False)
                self.log_info("process", f"Saved {filename} to dimension comparison directory: Dimensions match")
                matched_files.append(filename)
                
            except Exception as e:
                results[filename] = (False, f"Error comparing dimensions: {str(e)}")
                # Save to flagged directory on error
                target_file = self.flagged_dir / f"{filename}.csv"
                ocr_df.to_csv(target_file, index=False)
                self.log_error("process", f"Saved {filename} to flagged directory: Error during comparison")
                mismatched_files.append(filename)
        
        # Include unmatched files in the mismatched_files list in the results
        mismatched_files.extend(unmatched_ocr_files)
        
        # Return summary of results
        return {
            "dimension_comparison_status": "completed",
            "matched_files": matched_files,
            "mismatched_files": mismatched_files,
            "total_files": len(ocr_data),
            "matched_count": len(matched_files),
            "mismatched_count": len(mismatched_files),
            "dimension_comparison_results": results,
            "dimension_comparison_dir": str(self.checkpoint_dir)
        } 
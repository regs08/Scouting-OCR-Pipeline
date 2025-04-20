from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd
import re
import os
from .base_comparison_processor import BaseComparisonProcessor

class ColumnProcessor(BaseComparisonProcessor):
    """
    Processor to handle column mismatches by populating static columns from ground truth files.
    """
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 source_checkpoint_name: str = "ckpt2_dimension_comparison",
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the column processor.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            source_checkpoint_name: Name of the source checkpoint to use as input
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            source_checkpoint_name=source_checkpoint_name,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "column_processor"
        )
        
        # Create checkpoint directory for column processing
        self.checkpoint_dir = self.path_manager.get_checkpoint_path(
            session_id=self.session_id,
            checkpoint="ckpt3_column_correction"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create flagged directory for problematic files
        self.flagged_dir = self.path_manager.get_flagged_path(
            session_id=self.session_id,
            reason="column_issues"
        )
        self.flagged_dir.mkdir(parents=True, exist_ok=True)
        
    
    def _validate_dataframes(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[bool, str, Dict[str, List[Tuple[str, int]]]]:
        """
        Validate that both DataFrames have the correct structure.
        
        Args:
            gt_df: Ground truth DataFrame
            pred_df: Prediction DataFrame
            
        Returns:
            Tuple of (is_valid, message, column_info)
        """
        # Get all columns from both DataFrames
        gt_cols = set(gt_df.columns)
        pred_cols = set(pred_df.columns)
        
        # Find extra and missing columns
        extra_cols = [(col, pred_df.columns.get_loc(col)) 
                     for col in pred_cols - gt_cols]
        missing_cols = [(col, gt_df.columns.get_loc(col)) 
                       for col in gt_cols - pred_cols]
        
        # Build detailed message
        message_parts = []
        
        if extra_cols:
            message_parts.append(f"➕ Extra columns in prediction: {[f'{col} (at index {idx})' for col, idx in extra_cols]}")
            
        if missing_cols:
            message_parts.append(f"➖ Missing columns in prediction: {[f'{col} (at index {idx})' for col, idx in missing_cols]}")
            
        if not extra_cols and not missing_cols and len(gt_cols) == len(pred_cols):
            message_parts.append("✅ Columns match exactly")
            
        # Define message and is_valid based on validation results    
        message = " | ".join(message_parts) if message_parts else "No column differences found"
        is_valid = not (extra_cols or missing_cols)
        
        # Create column info dictionary
        column_info = {
            'extra_columns': extra_cols,
            'missing_columns': missing_cols
        }
            
        return is_valid, message, column_info
    
    def _populate_missing_columns(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix column names in prediction DataFrame to match ground truth DataFrame based on position.
        This method only renames columns without modifying any data values.
        
        Args:
            gt_df: Ground truth DataFrame
            pred_df: Prediction DataFrame
            
        Returns:
            Updated prediction DataFrame with corrected column names
        """
        # Create a copy of prediction DataFrame to keep all original values
        processed_df = pred_df.copy()
        
        # Get lists of column names
        gt_columns = list(gt_df.columns)
        pred_columns = list(processed_df.columns)
        
        # Check if column count matches
        if len(gt_columns) != len(pred_columns):
            self.log_warning("_populate_missing_columns", 
                           f"Column count mismatch: GT={len(gt_columns)}, Pred={len(pred_columns)}")
            
            # Determine the smaller length to avoid index errors
            min_cols = min(len(gt_columns), len(pred_columns))
            
            # Rename only the columns that exist in both DataFrames based on position
            rename_dict = {pred_columns[i]: gt_columns[i] for i in range(min_cols)}
            
            # If prediction has more columns than ground truth, keep those extra columns
            if len(pred_columns) > len(gt_columns):
                for i in range(min_cols, len(pred_columns)):
                    rename_dict[pred_columns[i]] = pred_columns[i]  # Keep original name
        else:
            # If column counts match, create rename dictionary based on position
            rename_dict = {pred_columns[i]: gt_columns[i] for i in range(len(pred_columns))}
        
        # Log the rename operations
        for old_name, new_name in rename_dict.items():
            if old_name != new_name:
                self.log_info("_populate_missing_columns", f"Renaming column '{old_name}' to '{new_name}'")
        
        # Rename columns
        processed_df = processed_df.rename(columns=rename_dict)
        
        return processed_df
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process prediction files to populate missing static columns from ground truth.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Updated dictionary with column processing results
        """
        # Get session directory
        session_dir = input_data.get('session_dir')
        if not session_dir:
            raise ValueError("Missing session_dir in input data")
            
        session_dir = Path(session_dir)
        
        # Get input directory (from source checkpoint)
        input_dir = None
        
        # If source_checkpoint_name is provided, use it to find the input directory
        if self.source_checkpoint_name:
            source_checkpoint_key = f"{self.source_checkpoint_name}_dir"
            if source_checkpoint_key in input_data:
                input_dir = input_data.get(source_checkpoint_key)
            else:
                # Try finding it without the _dir suffix
                input_dir = input_data.get(self.source_checkpoint_name)
        
        # Fallback to dimension_comparison_dir for backward compatibility
        if not input_dir:
            input_dir = input_data.get('dimension_comparison_dir')
        
        if not input_dir:
            self.log_error("process", f"No input directory found for source checkpoint: {self.source_checkpoint_name}")
            return {"column_processor_status": "error", "error": "No input directory found"}
        
        input_dir = Path(input_dir)
        
        # Load ground truth and prediction data
        gt_data = self._load_ground_truth(session_dir)
        pred_data = self._load_dataframes_from_dir(input_dir, "prediction")
        
        if not gt_data:
            self.log_error("process", "No ground truth data found")
            # Move all prediction files to flagged directory
            for filename, df in pred_data.items():
                target_file = self.flagged_dir / f"{filename}.csv"
                df.to_csv(target_file, index=False)
                self.log_warning("process", f"Saved {filename} to flagged directory: No ground truth data available")
            
            return {
                "column_processor_status": "completed",
                "matched_files": [],
                "flagged_files": list(pred_data.keys()),
                "total_files": len(pred_data),
                "matched_count": 0,
                "flagged_count": len(pred_data),
                "column_processor_dir": str(self.checkpoint_dir)
            }
        
        if not pred_data:
            self.log_error("process", "No prediction data found")
            return {"column_processor_status": "error", "error": "No prediction data found"}
        
        # Find matching files
        matching_files = self._find_matching_files(gt_data, pred_data)
        
        # Handle files that exist in prediction data but have no matching ground truth
        unmatched_pred_files = set(pred_data.keys()) - set(matching_files.keys())
        for filename in unmatched_pred_files:
            target_file = self.flagged_dir / f"{filename}.csv"
            pred_data[filename].to_csv(target_file, index=False)
            self.log_warning("process", f"Saved {filename} to flagged directory: No matching ground truth file")
        
        # Process matching files
        processed_files = []
        flagged_files = list(unmatched_pred_files)
        column_changes = {}
        
        for filename, data_pair in matching_files.items():
            try:
                gt_df = data_pair['gt']
                pred_df = data_pair['pred']
                
                # Validate columns
                is_valid, message, column_info = self._validate_dataframes(gt_df, pred_df)
                
                if self.verbose:
                    self.log_info("process", f"\n=== Column Validation for {filename} ===")
                    self.log_info("process", message)
                
                # Process if there are missing columns
                if not is_valid and column_info['missing_columns']:
                    # Populate missing columns
                    processed_df = self._populate_missing_columns(gt_df, pred_df)
                    
                    # Save to checkpoint directory
                    target_file = self.checkpoint_dir / f"{filename}.csv"
                    processed_df.to_csv(target_file, index=False)
                    self.log_info("process", f"Saved {filename} to column processor directory with fixed columns")
                    
                    processed_files.append(filename)
                    column_changes[filename] = {
                        'extra_columns': [col for col, _ in column_info['extra_columns']],
                        'missing_columns': [col for col, _ in column_info['missing_columns']],
                        'message': message
                    }
                elif is_valid:
                    # If already valid, just copy to checkpoint directory
                    target_file = self.checkpoint_dir / f"{filename}.csv"
                    pred_df.to_csv(target_file, index=False)
                    self.log_info("process", f"Saved {filename} to column processor directory (already valid)")
                    
                    processed_files.append(filename)
                    column_changes[filename] = {
                        'extra_columns': [],
                        'missing_columns': [],
                        'message': message
                    }
                else:
                    # If there are only extra columns but no missing ones, copy but log warning
                    target_file = self.checkpoint_dir / f"{filename}.csv"
                    pred_df.to_csv(target_file, index=False)
                    self.log_warning("process", 
                                   f"Saved {filename} to column processor directory with extra columns (not removed)")
                    
                    processed_files.append(filename)
                    column_changes[filename] = {
                        'extra_columns': [col for col, _ in column_info['extra_columns']],
                        'missing_columns': [],
                        'message': message
                    }
            except Exception as e:
                self.log_error("process", f"Error processing {filename}: {str(e)}")
                # Save to flagged directory on error
                target_file = self.flagged_dir / f"{filename}.csv"
                data_pair['pred'].to_csv(target_file, index=False)
                self.log_error("process", f"Saved {filename} to flagged directory: Error during processing")
                flagged_files.append(filename)
        
        # Return summary of results
        return {
            "column_processor_status": "completed",
            "matched_files": processed_files,
            "flagged_files": flagged_files,
            "total_files": len(pred_data),
            "matched_count": len(processed_files),
            "flagged_count": len(flagged_files),
            "column_changes": column_changes,
            "column_processor_dir": str(self.checkpoint_dir)
        } 
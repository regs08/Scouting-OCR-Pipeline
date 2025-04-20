from typing import Dict, Optional, Any
from pathlib import Path
import pandas as pd
import re

from .base_session_processor import BaseSessionProcessor

class BaseComparisonProcessor(BaseSessionProcessor):
    """Base class for session processors that compare data with ground truth."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 source_checkpoint_name: str = None,
                 case_sensitive: bool = False,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the base comparison processor.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            source_checkpoint_name: Name of the source checkpoint to compare against ground truth
            case_sensitive: Whether to perform case-sensitive comparisons
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
            operation_name=operation_name
        )
        
        self.source_checkpoint_name = source_checkpoint_name
        self.case_sensitive = case_sensitive
    
    def _load_dataframes_from_dir(self, directory: Path, description: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load DataFrames from CSV files in a directory.
        
        Args:
            directory: Path to the directory containing CSV files
            description: Description of the data being loaded (for logging)
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        if not directory.exists():
            self.log_warning("_load_dataframes_from_dir", f"Directory not found: {directory}")
            return {}
            
        data_dict = {}
        for file_path in directory.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                data_dict[file_path.stem] = df
                self.log_info("_load_dataframes_from_dir", f"Loaded {description} from {file_path.name}")
            except Exception as e:
                self.log_error("_load_dataframes_from_dir", f"Error loading {file_path.name}: {str(e)}")
                
        return data_dict
    
    def _load_ground_truth(self, session_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load ground truth data from the session directory.
        
        Args:
            session_dir: Path to the session directory
            
        Returns:
            Dictionary mapping filenames to ground truth DataFrames
        """
        gt_dir = session_dir / "ground_truth"
        return self._load_dataframes_from_dir(gt_dir, "ground truth data")
    
    def _load_checkpoint_data(self, checkpoint_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load data from the specified checkpoint directory.
        
        Args:
            checkpoint_dir: Path to the checkpoint directory
            
        Returns:
            Dictionary mapping filenames to checkpoint DataFrames
        """
        return self._load_dataframes_from_dir(checkpoint_dir, "checkpoint data")
    
    def _find_matching_files(self, gt_data: Dict[str, pd.DataFrame], 
                            checkpoint_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Find matching files between ground truth and checkpoint data based on R#P# pattern.
        
        Args:
            gt_data: Dictionary of ground truth DataFrames
            checkpoint_data: Dictionary of checkpoint DataFrames
            
        Returns:
            Dictionary of matching files with their corresponding DataFrames
        """
        matching_files = {}
        # Pattern to match R#P# where # is a digit (e.g., R1P1, R10P2, etc.)
        pattern = r'R\d+P\d+(?:_R\d+P\d+)*'
        
        for ckpt_file, ckpt_df in checkpoint_data.items():
            # Extract R#P# pattern from checkpoint filename
            ckpt_matches = re.findall(pattern, ckpt_file)
            if not ckpt_matches:
                self.log_warning("_find_matching_files", f"Skipping {ckpt_file}: No R#P# pattern found")
                continue
            
            ckpt_pattern = ckpt_matches[0]  # Use the first match
            
            # Look for matching ground truth file with the same pattern
            matching_gt = None
            for gt_file, gt_df in gt_data.items():
                gt_matches = re.findall(pattern, gt_file)
                if gt_matches and gt_matches[0] == ckpt_pattern:
                    matching_gt = gt_file
                    matching_files[ckpt_file] = {
                        'pred': ckpt_df,
                        'gt': gt_df
                    }
                    self.log_info("_find_matching_files", 
                                f"Matched {ckpt_file} with ground truth {gt_file} using pattern {ckpt_pattern}")
                    break
                
            if not matching_gt:
                self.log_warning("_find_matching_files", 
                               f"No matching ground truth file found for {ckpt_file} with pattern {ckpt_pattern}")
            
        return matching_files
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process comparison between checkpoint data and ground truth.
        This method must be implemented by subclasses.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Dictionary with processing results
        """
        raise NotImplementedError("Subclasses must implement process") 
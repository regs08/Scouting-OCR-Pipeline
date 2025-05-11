import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import re

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent
from utils.data_classes import MatchedFile


class MatchGTAndFileComponent(PipelineComponent):
    """
    Component that finds matching files between ground_truth and a comparison directory.
    Moves any unmatched files to the flagged/unmatched directory.
    For GT files, removes the '_gt_' part before comparing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_manager = None
        self.session_id = None
        self.gt_dir = None
        self.compare_dir = None
        self.flagged_dir = None
        self.unmatched_dir = None

    def _get_gt_directory(self, session_paths: Dict[str, Path]) -> Path:
        """
        Helper method to get the ground truth directory.
        
        Args:
            session_paths: Dictionary of session paths
            
        Returns:
            Path to ground truth directory
            
        Raises:
            ValueError: If ground truth directory is not found
        """
        gt_dir = session_paths.get('ground_truth')
        if not gt_dir:
            raise ValueError("ground_truth directory must exist in session paths")
        return gt_dir

    def _get_compare_directory(self, input_data: Dict[str, Any], session_paths: Dict[str, Path]) -> Path:
        """
        Helper method to get the comparison directory.
        Priority:
        1. Explicit compare_dir in input_data
        2. compare_dir_key in input_data
        3. Default to 'raw' directory
        
        Args:
            input_data: Input data dictionary
            session_paths: Dictionary of session paths
            
        Returns:
            Path to comparison directory
            
        Raises:
            ValueError: If comparison directory is not found
        """
        # Check for explicit compare_dir first
        if 'compare_dir' in input_data:
            return Path(input_data['compare_dir'])
            
        # Use compare_dir_key or default to 'raw'
        else: 
            compare_dir_key = input_data.get('compare_dir_key', 'raw')
            compare_dir = session_paths.get(compare_dir_key)
        
        if not compare_dir:
            raise ValueError(f"Comparison directory '{compare_dir_key}' must exist in session paths")
            
        return compare_dir

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare directories for matching. Allows comparison of GT folder to a configurable folder.
        
        Args:
            input_data: Input data dictionary containing:
                - path_manager: PathManager instance
                - session_id: Session identifier
                - compare_dir_key: Optional key for comparison directory (default: 'raw')
                - compare_dir: Optional explicit path to comparison directory
                - checkpoint_name: Optional name for unmatched directory (default: 'unmatched')
                
        Returns:
            Updated input data dictionary
            
        Raises:
            ValueError: If required directories or data are missing
        """
        # Get required path manager and session ID
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
            
        # Get session paths
        session_paths = self.path_manager.get_session_paths(self.session_id)
        
        # Get ground truth directory
        self.gt_dir = self._get_gt_directory(session_paths)
        
        # Get comparison directory
        self.compare_dir = self._get_compare_directory(input_data, session_paths)
        
        # Get flagged directory
        self.flagged_dir = session_paths.get('flagged')
        if not self.flagged_dir:
            raise ValueError("flagged directory must exist in session paths")
            
        # Create unmatched directory
        checkpoint_name = input_data.get('checkpoint_name', 'unmatched')
        self.unmatched_dir = self.flagged_dir / checkpoint_name
        self.unmatched_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_info("process_before_pipeline", "Directories prepared for matching", {
            "gt_dir": str(self.gt_dir),
            "compare_dir": str(self.compare_dir),
            "unmatched_dir": str(self.unmatched_dir)
        })
        
        return input_data

    def _normalize_gt_filename(self, filename: str) -> str:
        """
        Normalize filename by:
        1. Removing extension
        2. Removing '_gt_' from the filename
        3. Removing any remaining '_gt' suffix
        
        Example: 
        - DML_gt_20241408_R1T6.pdf -> DML_20241408_R1T6
        - DML_20241408_R1T6_gt.csv -> DML_20241408_R1T6
        """
        # Remove extension first
        base_name = Path(filename).stem
        # Remove '_gt_' from middle of filename
        normalized = re.sub(r'_gt_', '_', base_name, count=1)
        # Remove '_gt' from end of filename if present
        normalized = re.sub(r'_gt$', '', normalized)
        return normalized

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare files in the GT folder to the comparison folder.
        Moves any unmatched files to the flagged/unmatched directory.
        
        Args:
            pipeline_output: Output data from previous pipeline steps
            
        Returns:
            Updated pipeline output with matching results
        """
        # List all files in compare_dir and ground_truth
        compare_files = {self._normalize_gt_filename(f.name): f.name for f in self.compare_dir.iterdir() if f.is_file()}
        gt_files = {self._normalize_gt_filename(f.name): f.name for f in self.gt_dir.iterdir() if f.is_file()}
        
        # Find unmatched files
        unmatched_compare = [compare_files[norm] for norm in compare_files.keys() if norm not in gt_files]
        unmatched_gt = [gt_files[norm] for norm in gt_files.keys() if norm not in compare_files]
        
        moved = []
        # Move unmatched files from compare_dir
        for file_name in unmatched_compare:
            src = self.compare_dir / file_name
            dest = self.unmatched_dir / file_name
            try:
                shutil.move(str(src), str(dest))
                moved.append({'type': 'compare', 'name': file_name, 'from': str(src), 'to': str(dest)})
                self.log_info("process_after_pipeline", f"Moved unmatched comparison file to flagged/unmatched", {
                    'file': file_name,
                    'from': str(src),
                    'to': str(dest)
                })
            except Exception as e:
                self.log_error("process_after_pipeline", f"Failed to move unmatched comparison file", {
                    'file': file_name,
                    'from': str(src),
                    'to': str(dest),
                    'error': str(e)
                })
                
        # Move unmatched gt files
        for file_name in unmatched_gt:
            src = self.gt_dir / file_name
            dest = self.unmatched_dir / file_name
            try:
                shutil.move(str(src), str(dest))
                moved.append({'type': 'gt', 'name': file_name, 'from': str(src), 'to': str(dest)})
                self.log_info("process_after_pipeline", f"Moved unmatched gt file to flagged/unmatched", {
                    'file': file_name,
                    'from': str(src),
                    'to': str(dest)
                })
            except Exception as e:
                self.log_error("process_after_pipeline", f"Failed to move unmatched gt file", {
                    'file': file_name,
                    'from': str(src),
                    'to': str(dest),
                    'error': str(e)
                })
        
        # Create MatchedFile objects for matched pairs
        matched_files = []
        for norm_name in set(compare_files.keys()) & set(gt_files.keys()):
            compare_file = self.compare_dir / compare_files[norm_name]
            gt_file = self.gt_dir / gt_files[norm_name]
            if compare_file.exists() and gt_file.exists():
                matched_file = MatchedFile(
                    normalized_name=norm_name,
                    pred_path=compare_file,
                    gt_path=gt_file
                )
                matched_files.append(matched_file)
                self.log_info("process_after_pipeline", "Found matched file pair", {
                    'normalized_name': norm_name,
                    'compare_file': str(compare_file),
                    'gt_file': str(gt_file)
                })
        
        self.log_info("process_after_pipeline", "File matching completed", {
            'num_matched': len(matched_files),
            'num_unmatched_compare': len(unmatched_compare),
            'num_unmatched_gt': len(unmatched_gt)
        })
        
        return {
            **pipeline_output,
            'matched_files': matched_files,
            'unmatched_moved': moved,
            'unmatched_dir': str(self.unmatched_dir)
        } 
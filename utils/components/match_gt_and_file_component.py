import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil
import re
from dataclasses import dataclass, asdict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent

@dataclass(frozen=True)
class MatchedFile:
    gt_path: str
    compare_path: str
    normalized_name: str

class MatchGTAndFileComponent(PipelineComponent):
    """
    Component that finds matching files between raw and ground_truth directories.
    Moves any unmatched files to the flagged/unmatched directory.
    For GT files, removes the '_gt_' part before comparing.
    """
    def __init__(self, *args, compare_dir_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_dir_key = compare_dir_key
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare directories for matching. Allows comparison of GT folder to a configurable folder (default: 'raw').
        Set 'compare_dir_key' in input_data to override the default.
        """
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
        session_paths = self.path_manager.get_session_paths(self.session_id)
        self.gt_dir = session_paths.get('ground_truth')
        self.flagged_dir = session_paths.get('flagged')
        if not self.gt_dir or not self.flagged_dir:
            raise ValueError("ground_truth and flagged directories must exist in session paths")
        # Use the instance attribute for compare_dir_key
        # very hacky way to get the compare_dir_key from the ocr_output_dir
        if not self.compare_dir_key:
            self.compare_dir_key = 'raw'
        if 'ocr_output_dir' in input_data:
            self.compare_dir = Path(input_data['ocr_output_dir'])
        else:
            self.compare_dir = session_paths.get(self.compare_dir_key)        
 
        if not self.compare_dir:
            raise ValueError(f"{self.compare_dir_key} directory must exist in session paths")

        self.unmatched_dir = self.flagged_dir / input_data.get('checkpoint_name', 'unmatched')
        self.unmatched_dir.mkdir(parents=True, exist_ok=True)
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
        Compare files in the GT folder to the selected comparison folder (default: raw).
        Moves any unmatched files to the flagged/unmatched directory.
        """
        # List all files in compare_dir and ground_truth
        compare_files = {self._normalize_gt_filename(f.name): f.name for f in self.compare_dir.iterdir() if f.is_file()}
        gt_files = {self._normalize_gt_filename(f.name): f.name for f in self.gt_dir.iterdir() if f.is_file()}
        
        # Find unmatched in compare_dir (no corresponding GT file)
        unmatched_compare = [compare_files[norm] for norm in compare_files.keys() if norm not in gt_files]
        # Find unmatched in GT (no corresponding file in compare_dir)
        unmatched_gt = [gt_files[norm] for norm in gt_files.keys() if norm not in compare_files]
        
        moved = []
        # Move unmatched files from compare_dir
        for file_name in unmatched_compare:
            src = self.compare_dir / file_name
            dest = self.unmatched_dir / file_name
            try:
                shutil.move(str(src), str(dest))
                moved.append({'type': self.compare_dir_key, 'name': file_name, 'from': str(src), 'to': str(dest)})
                self.log_info("process_after_pipeline", f"Moved unmatched {self.compare_dir_key} file to flagged/unmatched", {
                    'file': file_name,
                    'from': str(src),
                    'to': str(dest)
                })
            except Exception as e:
                self.log_error("process_after_pipeline", f"Failed to move unmatched {self.compare_dir_key} file", {
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
        # Find matched files (normalized names present in both)
        matched_keys = set(compare_files.keys()) & set(gt_files.keys())
        matched_files = [
            MatchedFile(
                gt_path=str(self.gt_dir / gt_files[key]),
                compare_path=str(self.compare_dir / compare_files[key]),
                normalized_name=key
            )
            for key in matched_keys
        ]

        output = {
            **pipeline_output,
            'unmatched_moved': moved,
            'unmatched_dir': str(self.unmatched_dir),
            'matched_files': matched_files
        }
        return output 
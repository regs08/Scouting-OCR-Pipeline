import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil
import re

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent

class MatchGTAndRawComponent(PipelineComponent):
    """
    Component that finds matching files between raw and ground_truth directories.
    Moves any unmatched files to the flagged/unmatched directory.
    For GT files, removes the '_gt_' part before comparing.
    """
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
        session_paths = self.path_manager.get_session_paths(self.session_id)
        self.raw_dir = session_paths.get('raw')
        self.gt_dir = session_paths.get('ground_truth')
        self.flagged_dir = session_paths.get('flagged')
        if not self.raw_dir or not self.gt_dir or not self.flagged_dir:
            raise ValueError("raw, ground_truth, and flagged directories must exist in session paths")
        self.unmatched_dir = self.flagged_dir / 'unmatched'
        self.unmatched_dir.mkdir(parents=True, exist_ok=True)
        return input_data

    def _normalize_gt_filename(self, filename: str) -> str:
        # Remove '_gt_' from the filename for matching
        # Example: DML_gt_20241408_R1T6.pdf -> DML_20241408_R1T6.pdf
        return re.sub(r'_gt_', '_', filename, count=1)

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        # List all files in raw and ground_truth
        raw_files = {f.name for f in self.raw_dir.iterdir() if f.is_file()}
        gt_files = {f.name for f in self.gt_dir.iterdir() if f.is_file()}
        # Normalize GT filenames for matching
        normalized_gt_files = {self._normalize_gt_filename(f): f for f in gt_files}
        # Find unmatched in raw (no corresponding GT file)
        unmatched_raw = [f for f in raw_files if f not in normalized_gt_files]
        # Find unmatched in GT (no corresponding raw file)
        normalized_raw_files = {self._normalize_gt_filename(f): f for f in raw_files}
        unmatched_gt = [f for norm, f in normalized_gt_files.items() if norm not in raw_files]
        moved = []
        # Move unmatched raw files
        for file_name in unmatched_raw:
            src = self.raw_dir / file_name
            dest = self.unmatched_dir / file_name
            try:
                shutil.move(str(src), str(dest))
                moved.append({'type': 'raw', 'name': file_name, 'from': str(src), 'to': str(dest)})
                self.log_info("process_after_pipeline", f"Moved unmatched raw file to flagged/unmatched", {
                    'file': file_name,
                    'from': str(src),
                    'to': str(dest)
                })
            except Exception as e:
                self.log_error("process_after_pipeline", f"Failed to move unmatched raw file", {
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
        output = {
            **pipeline_output,
            'unmatched_moved': moved,
            'unmatched_dir': str(self.unmatched_dir)
        }
        return output 
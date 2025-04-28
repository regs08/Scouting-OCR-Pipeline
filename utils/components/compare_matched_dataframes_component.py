import sys
from pathlib import Path
from typing import Dict, Any
import shutil
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent

class CompareMatchedDataFramesComponent(PipelineComponent):
    """
    Component that loads matched file pairs as DataFrames and copies them to the checkpoint folder.
    No comparison is performed yet; this is a preparatory step for further processing.
    """
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates loading and copying of matched file pairs. Uses helpers for checkpoint dir and file operations.
        """
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        self.checkpoint_name = input_data.get('checkpoint_name', 'compare_matched')
        matched_files = input_data.get('matched_files', [])
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
        if not matched_files:
            self.log_info("process_before_pipeline", "No matched files to process.")
            return input_data
        checkpoint_dir = self._get_checkpoint_dir()
        self.log_info("process_before_pipeline", f"Copying matched files to checkpoint folder {checkpoint_dir}")
        loaded_pairs = []
        for pair in matched_files:
            loaded = self._load_and_copy_pair(pair, checkpoint_dir)
            if loaded:
                loaded_pairs.append(loaded)
        input_data['loaded_pairs'] = loaded_pairs
        input_data['checkpoint_dir'] = str(checkpoint_dir)
        return input_data

    def _get_checkpoint_dir(self) -> Path:
        """
        Returns the checkpoint directory for this component, creating it if necessary.
        """
        session_paths = self.path_manager.get_session_paths(self.session_id)
        checkpoint_dir = session_paths.get(self.checkpoint_name)
        if not checkpoint_dir:
            checkpoint_dir = session_paths.get('flagged', Path('.')) / self.checkpoint_name
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def _load_and_copy_pair(self, pair: dict, checkpoint_dir: Path) -> dict:
        """
        Loads the GT and comparison files as DataFrames and copies them to the checkpoint directory.
        Returns a dict with the new file paths and normalized name, or None if failed.
        """
        gt_path = Path(pair['gt_path'])
        compare_path = Path(pair['compare_path'])
        try:
            gt_df = pd.read_csv(gt_path)
            compare_df = pd.read_csv(compare_path)
            gt_dest = checkpoint_dir / f"gt_{gt_path.name}"
            compare_dest = checkpoint_dir / f"compare_{compare_path.name}"
            shutil.copy2(gt_path, gt_dest)
            shutil.copy2(compare_path, compare_dest)
            self.log_info("_load_and_copy_pair", f"Copied pair to checkpoint folder", {
                'gt': str(gt_dest),
                'compare': str(compare_dest)
            })
            return {
                'gt_path': str(gt_dest),
                'compare_path': str(compare_dest),
                'normalized_name': pair['normalized_name']
            }
        except Exception as e:
            self.log_error("_load_and_copy_pair", f"Failed to load or copy matched files", {
                'gt_path': str(gt_path),
                'compare_path': str(compare_path),
                'error': str(e)
            })
            return None 
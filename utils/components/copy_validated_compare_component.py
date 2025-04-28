import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil
import pandas as pd
from dataclasses import dataclass

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent

@dataclass(frozen=True)
class MatchedFile:
    gt_path: str
    compare_path: str
    normalized_name: str

class CopyValidatedCompareComponent(PipelineComponent):
    """
    Component that loads matched CSV file pairs as DataFrames.
    If both load successfully, copies the compare file to the checkpoint folder.
    """

    @staticmethod
    def dataframes_dimensions_match(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Returns True if both DataFrames have the same shape (rows and columns).
        """
        return df1.shape == df2.shape

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        self.matched_files = input_data.get('matched_files')
        # if not self.matched_files:
        #     raise ValueError("matched_files are required in input_data")
        if not self.path_manager or not self.session_id :
            raise ValueError("path_manager and session_id are required in input_data")
        return input_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads each matched file pair as DataFrames.
        If both load successfully, copies the compare file to the checkpoint folder.
        If dimensions do not match, moves the compare file to a 'dimension_mismatch' folder under flagged.
        """
        self.log_info("process_after_pipeline", f"Matched files to process: {len(self.matched_files)}")
        # Use PathManager's method to get the correct checkpoint folder
        checkpoint_name = getattr(self, 'operation_name', None) or f"ckpt{self.checkpoint_number}"
        checkpoint_folder = self.path_manager.get_checkpoint_path(self.session_id, checkpoint_name)
        checkpoint_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Create dimension mismatch folder under flagged
        dimension_mismatch_dir = self.path_manager.get_flagged_path(self.session_id, 'dimension_mismatch')
        dimension_mismatch_dir.mkdir(parents=True, exist_ok=True)

        self.log_info("process_after_pipeline", f"Checkpoint folder: {checkpoint_folder}")
        copied = []
        dimension_mismatches = []
        for mf in self.matched_files:
            try:
                gt_df = pd.read_csv(mf.gt_path)
                compare_df = pd.read_csv(mf.compare_path)
                # Remove the first row from the compare DataFrame
                compare_df = compare_df.iloc[1:].reset_index(drop=True)
                if self.dataframes_dimensions_match(gt_df, compare_df):
                    # Rename columns to match GT
                    compare_df.columns = gt_df.columns
                    dest = checkpoint_folder / Path(mf.compare_path).name
                    # Save the modified DataFrame (with first row dropped and columns renamed)
                    compare_df.to_csv(dest, index=False)
                    copied.append({
                        'normalized_name': mf.normalized_name,
                        'compare_path': mf.compare_path,
                        'copied_to': str(dest)
                    })
                    self.log_info("process_after_pipeline", f"Copied validated compare file", {
                        'compare_path': mf.compare_path,
                        'dest': str(dest)
                    })
                else:
                    # Copy to dimension mismatch folder (do not move)
                    dest = dimension_mismatch_dir / Path(mf.compare_path).name
                    shutil.copy2(mf.compare_path, dest)
                    dimension_mismatches.append({
                        'normalized_name': mf.normalized_name,
                        'compare_path': mf.compare_path,
                        'copied_to': str(dest)
                    })
                    self.log_error("process_after_pipeline", f"Dimension mismatch: copied compare file to dimension_mismatch", {
                        'compare_path': mf.compare_path,
                        'dest': str(dest),
                        'gt_shape': gt_df.shape,
                        'compare_shape': compare_df.shape
                    })
            except Exception as e:
                self.log_error("process_after_pipeline", f"Failed to load or copy/move matched files", {
                    'gt_path': mf.gt_path,
                    'compare_path': mf.compare_path,
                    'error': str(e)
                })

        output = {
            **pipeline_output,
            'copied_validated_compare': copied,
            'dimension_mismatches': dimension_mismatches,
            'checkpoint_folder': str(checkpoint_folder),
            'dimension_mismatch_dir': str(dimension_mismatch_dir)
        }

        # Print a readable report of dimension mismatches
        if dimension_mismatches:
            print("\n=== Dimension Mismatch Report ===")
            for mismatch in dimension_mismatches:
                print(f"File: {mismatch['compare_path']}")
                print(f"  Copied to: {mismatch['copied_to']}")
                print(f"  GT shape: {mismatch.get('gt_shape', 'N/A')}")
                print(f"  Compare shape: {mismatch.get('compare_shape', 'N/A')}")
                print("-----------------------------")
            print("=== End of Dimension Mismatch Report ===\n")

        return output 
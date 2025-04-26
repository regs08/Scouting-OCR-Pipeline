import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent

class CopyRawFilesToSessionComponent(PipelineComponent):
    """
    Pipeline component that receives validated files, logs them, and copies them to the session's raw directory.
    """
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        valid_files: List[Path] = input_data.get('valid_files', [])
        session_dir = None
        raw_dir = None
        # Try to get the session/raw directory from path_manager if available
        path_manager = input_data.get('path_manager')
        if path_manager:
            # path_manager.base_dir / session_id / 'raw'
            session_id = input_data.get('session_id')
            if session_id:
                session_dir = path_manager.base_dir / session_id
                raw_dir = session_dir / 'raw'
                raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_info("process_before_pipeline", f"Received {len(valid_files)} valid files", {
            "valid_files": [str(f) for f in valid_files],
            "raw_dir": str(raw_dir) if raw_dir else None
        })
        
        if raw_dir:
            for file_path in valid_files:
                dest_path = raw_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_path)
                    self.log_info("process_before_pipeline", f"Copied file to session raw dir", {
                        "source": str(file_path),
                        "destination": str(dest_path)
                    })
                except Exception as e:
                    self.log_error("process_before_pipeline", f"Failed to copy file", {
                        "source": str(file_path),
                        "destination": str(dest_path),
                        "error": str(e)
                    })
        else:
            self.log_warning("process_before_pipeline", "No session/raw directory found, skipping copy.")
        return input_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        # Just return the output, don't reference self.path_manager
        return pipeline_output 
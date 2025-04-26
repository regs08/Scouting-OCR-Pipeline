import sys
from pathlib import Path
from typing import Dict, List, Union, Optional
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class PathManager:
    """Manages file paths for the OCR pipeline."""
    
    def __init__(self, expected_site_code: str, batch: str):
        """
        Initialize the path manager.
        
        Args:
            vineyard: Name of the vineyard
            batch: Batch identifier (date)
        """
        self.expected_site_code = Path(expected_site_code)
        self.batch = str(batch)
        self.base_dir = self.expected_site_code / self.batch
        
    def get_session_paths(self, session_id: str) -> Dict[str, Path]:
        """
        Returns all relevant paths for a session, including ground_truth.
        
        Args:
            session_id: Unique identifier for the session (timestamp)
            
        Returns:
            Dictionary of path types to their corresponding Path objects
        """
        session_base = self.base_dir / session_id
        ground_truth_dir = session_base / "ground_truth"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        return {
            'raw': session_base / "raw",
            'processed': session_base / "processed",
            'checkpoints': session_base / "processed" / "checkpoints",
            'flagged': session_base / "flagged",
            'logs': session_base / "logs",
            'ground_truth': ground_truth_dir
        }
        
    def get_checkpoint_path(self, session_id: str, checkpoint: str) -> Path:
        """
        Returns path for a specific checkpoint.
        
        Args:
            session_id: Unique identifier for the session
            checkpoint: Name of the checkpoint
            
        Returns:
            Path to the checkpoint directory
        """
        checkpoint_path = self.base_dir / session_id / "processed" / "checkpoints" / checkpoint
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path
        
    def get_flagged_path(self, session_id: str, reason: str) -> Path:
        """
        Returns path for flagged files.
        
        Args:
            session_id: Unique identifier for the session
            reason: Reason for flagging
            
        Returns:
            Path to the flagged files directory
        """
        flagged_path = self.base_dir / session_id / "flagged" / reason
        flagged_path.mkdir(parents=True, exist_ok=True)
        return flagged_path
        
    def get_log_path(self, session_id: str, log_type: str) -> Path:
        """
        Returns path for log files.
        
        Args:
            session_id: Unique identifier for the session
            log_type: Type of log (e.g., 'processing', 'errors')
            
        Returns:
            Path to the log directory
        """
        log_path = self.base_dir / session_id / "logs" / log_type
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    def get_panel_dir(self, panel_id: Union[str, int], category: str, stage: str = "raw") -> Path:
        """
        Get the directory for a specific panel's data.
        
        Args:
            panel_id: The panel identifier
            category: One of 'agents', 'original', 'groundtruth', 'ocr_predictions', 'cleaned', 'column_matched', 'checkpoints'
            stage: Either 'raw' or 'processed'
            
        Returns:
            Path to the panel's directory
        """
        base = self.base_dir / "raw" if stage == "raw" else self.base_dir / "processed"
        category_dir = base / category / "_by_id" / f"panel_{panel_id}"
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir
    
    def get_combined_dir(self, category: str, stage: str = "raw") -> Path:
        """Get the combined directory for a category."""
        base = self.base_dir / "raw" if stage == "raw" else self.base_dir / "processed"
        return base / category / "_combined"
    
    def get_file_path(self, panel_id: Union[str, int], filename: str, category: str, 
                      stage: str = "raw", combined: bool = False) -> Path:
        """
        Get the full path for a file.
        
        Args:
            panel_id: The panel identifier
            filename: Name of the file
            category: Category directory name
            stage: Either 'raw' or 'processed'
            combined: Whether to store in combined directory
            
        Returns:
            Complete file path
        """
        if combined:
            return self.get_combined_dir(category, stage) / filename
        return self.get_panel_dir(panel_id, category, stage) / filename
    
    def list_panels(self, category: str, stage: str = "raw") -> List[str]:
        """List all panel IDs in a category."""
        base = self.base_dir / "raw" if stage == "raw" else self.base_dir / "processed"
        panel_dir = base / category / "_by_id"
        if not panel_dir.exists():
            return []
        return [d.name.replace("panel_", "") for d in panel_dir.iterdir() if d.is_dir()]
    
    def ensure_panel_dirs(self, panel_id: Union[str, int]):
        """Ensures all necessary directories exist for a panel."""
        categories = ['agents', 'original', 'groundtruth', 'ocr_predictions', 'cleaned', 'column_matched', 'checkpoints']
        stages = ['raw', 'processed']
        
        for stage in stages:
            for category in categories:
                if stage == 'raw' and category in ['cleaned', 'column_matched', 'checkpoints']:
                    continue
                if stage == 'processed' and category in ['agents', 'original', 'groundtruth', 'ocr_predictions']:
                    continue
                self.get_panel_dir(panel_id, category, stage)

    def get_flagged_dir(self, session_id: str) -> Path:
        """
        Get the path for the flagged directory.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Path to the flagged directory
        """
        flagged_dir = self.base_dir / session_id / "flagged"
        flagged_dir.mkdir(parents=True, exist_ok=True)
        return flagged_dir

    def validate_setup(self) -> bool:
        """
        Validate that the basic directory structure exists.
        
        Returns:
            bool: True if setup is valid
        """
        # Check if base directory exists
        if not self.base_dir.exists():
            return False
            
        # Basic setup validation - check if essential directories exist
        essential_dirs = [
            self.base_dir,
            self.base_dir / "raw",
            self.base_dir / "processed"
        ]
        
        for directory in essential_dirs:
            if not directory.exists():
                return False
                
        return True 
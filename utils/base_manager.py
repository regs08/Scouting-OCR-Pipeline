import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor
from utils.path_manager import PathManager

class BaseManager(BaseProcessor):
    """Base class for managers that handle pipeline execution."""
    
    def __init__(self,
                 path_manager: PathManager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the base manager.
        
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
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "base_manager"
        )
        
        self.path_manager = path_manager
        self.session_id = session_id
        self.pipeline = []
        self.checkpoint_status = {}
        
    def add_processor(self, processor: BaseProcessor, checkpoint_name: str, checkpoint_number: int) -> None:
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Processor to add
            checkpoint_name: Name of the checkpoint
            checkpoint_number: Number of the checkpoint
        """
        self.pipeline.append({
            'processor': processor,
            'checkpoint_name': checkpoint_name,
            'checkpoint_number': checkpoint_number
        })
        self.checkpoint_status[checkpoint_name] = "pending"
        
    def _get_filtered_pipeline(self, start_checkpoint: Optional[int] = None, 
                             end_checkpoint: Optional[int] = None) -> List[Dict]:
        """
        Get the filtered pipeline based on checkpoint range.
        
        Args:
            start_checkpoint: Starting checkpoint number (inclusive)
            end_checkpoint: Ending checkpoint number (inclusive)
            
        Returns:
            Filtered list of processor information
        """
        if start_checkpoint is None and end_checkpoint is None:
            return self.pipeline
            
        filtered = []
        for processor_info in self.pipeline:
            checkpoint_number = processor_info['checkpoint_number']
            if (start_checkpoint is None or checkpoint_number >= start_checkpoint) and \
               (end_checkpoint is None or checkpoint_number <= end_checkpoint):
                filtered.append(processor_info)
                
        return filtered
        
    def get_checkpoint_status(self, checkpoint_name: str) -> str:
        """
        Get the status of a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Status of the checkpoint
        """
        return self.checkpoint_status.get(checkpoint_name, "unknown")
        
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        return [{
            'name': info['checkpoint_name'],
            'number': info['checkpoint_number'],
            'status': self.checkpoint_status[info['checkpoint_name']]
        } for info in self.pipeline] 
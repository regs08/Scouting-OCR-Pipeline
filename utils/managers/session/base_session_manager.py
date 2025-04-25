"""
Base class for session managers in the OCR pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from utils.managers.base_manager import BaseManager
from utils.path_manager import PathManager

class BaseSessionManager(BaseManager):
    """
    Base class for session managers that handle processing within a session.
    Provides common functionality for session-based operations.
    """
    
    def __init__(self,
                 path_manager: PathManager,
                 session_id: str,
                 expected_data_cols: List[str],
                 expected_index_cols: List[str],
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the base session manager.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            expected_data_cols: List of expected data column names
            expected_index_cols: List of expected index column names
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
            operation_name=operation_name or "session_manager"
        )
        
        self.expected_data_cols = expected_data_cols
        self.expected_index_cols = expected_index_cols
        
        # Initialize the pipeline
        self._setup_pipeline()
        
    def _setup_pipeline(self) -> None:
        """
        Set up the processing pipeline.
        This method should be overridden by child classes to add their specific processors.
        """
        raise NotImplementedError("Child classes must implement _setup_pipeline")
        
    def _get_previous_checkpoint_data(self, checkpoint_number: int) -> Dict[str, Any]:
        """
        Get data from a previous checkpoint.
        
        Args:
            checkpoint_number: Number of the checkpoint to retrieve data from
            
        Returns:
            Dictionary containing data from the previous checkpoint
        """
        # Get the checkpoint directory
        checkpoint_paths = self.path_manager.get_session_paths(self.session_id)
        checkpoint_dir = checkpoint_paths['checkpoints'] / f"ckpt{checkpoint_number}"
        
        if not checkpoint_dir.exists():
            self.log_warning("_get_previous_checkpoint_data", 
                           f"Checkpoint directory not found: {checkpoint_dir}")
            return {}
            
        data = {}
        csv_files = list(checkpoint_dir.glob("*.csv"))
        
        if not csv_files:
            self.log_warning("_get_previous_checkpoint_data", 
                           f"No CSV files found in {checkpoint_dir}")
            return {}
            
        for file_path in csv_files:
            try:
                data[file_path.stem] = self._load_checkpoint_file(file_path)
                self.log_info("_get_previous_checkpoint_data", 
                            f"Loaded data from {file_path.name}")
            except Exception as e:
                self.log_error("_get_previous_checkpoint_data", 
                             f"Error loading {file_path.name}: {str(e)}")
                continue
                
        return data
        
    def _load_checkpoint_file(self, file_path: Path) -> Any:
        """
        Load data from a checkpoint file.
        This method can be overridden by child classes to implement specific loading logic.
        
        Args:
            file_path: Path to the checkpoint file
            
        Returns:
            Loaded data
        """
        raise NotImplementedError("Child classes must implement _load_checkpoint_file")
        
    def prepare_pipeline_data(self) -> Dict[str, Any]:
        """
        Prepare the initial data for the pipeline.
        This method can be overridden by child classes to implement specific data preparation.
        
        Returns:
            Dictionary with initial pipeline data
        """
        # Get session directory from path manager
        session_paths = self.path_manager.get_session_paths(self.session_id)
        session_dir = self.path_manager.base_dir / self.session_id
            
        if not session_dir.exists():
            self.log_error("prepare_pipeline_data", f"Session directory not found: {session_dir}")
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        return {
            'session_id': self.session_id,
            'session_dir': session_dir,
            'session_paths': session_paths,
            'path_manager': self.path_manager,
            'expected_data_cols': self.expected_data_cols,
            'expected_index_cols': self.expected_index_cols
        }
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through the session pipeline.
        This method should be overridden by child classes to implement specific processing logic.
        
        Args:
            input_data: Dictionary containing input data for processing
            
        Returns:
            Dictionary containing processing results
        """
        raise NotImplementedError("Child classes must implement process")
        
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the session processing pipeline.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        try:
            self.log_info("run", f"Starting session processing for {self.session_id}")
            result = self.process(input_data)
            self.log_info("run", "Session processing completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error in session processing: {str(e)}"
            self.log_error("run", error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'checkpoint_status': self.checkpoint_status
            } 
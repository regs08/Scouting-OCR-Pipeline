from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from .base_processor import BaseProcessor
from .setup_processors.base_setup_processor import BaseSetupProcessor

class SetupPipelineExecutor(BaseProcessor):
    """Executes the setup pipeline for a session."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the setup pipeline executor.
        
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
            operation_name=operation_name or "setup_pipeline"
        )
        
        self.path_manager = path_manager
        self.session_id = session_id
        self.pipeline = []
        self.checkpoint_status = {}
        
    def add_processor(self, processor: BaseSetupProcessor, checkpoint_name: str, checkpoint_number: int):
        """
        Add a processor to the setup pipeline.
        
        Args:
            processor: Setup processor to add
            checkpoint_name: Name of the checkpoint
            checkpoint_number: Number of the checkpoint
        """
        self.pipeline.append({
            'processor': processor,
            'checkpoint_name': checkpoint_name,
            'checkpoint_number': checkpoint_number
        })
        self.checkpoint_status[checkpoint_name] = "pending"
        
    def execute(self, start_checkpoint: Optional[int] = None, end_checkpoint: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the setup pipeline.
        
        Args:
            start_checkpoint: Starting checkpoint number (inclusive)
            end_checkpoint: Ending checkpoint number (inclusive)
            
        Returns:
            Dictionary containing setup results
        """
        self.log_checkpoint("execute", "started", {
            "start_checkpoint": start_checkpoint,
            "end_checkpoint": end_checkpoint
        })
        
        try:
            # Initialize setup data
            setup_data = {
                'input_dir': str(self.path_manager.base_dir),
                'session_id': self.session_id
            }
            
            # Filter pipeline based on checkpoint range
            filtered_pipeline = self._get_filtered_pipeline(start_checkpoint, end_checkpoint)
            
            # Execute each processor in the pipeline
            for processor_info in filtered_pipeline:
                processor = processor_info['processor']
                checkpoint_name = processor_info['checkpoint_name']
                
                try:
                    # Process the data
                    setup_data = processor.process(setup_data)
                    
                    # Update checkpoint status
                    self.checkpoint_status[checkpoint_name] = "completed"
                    self.log_info("execute", f"Completed checkpoint: {checkpoint_name}")
                    
                except Exception as e:
                    self.checkpoint_status[checkpoint_name] = "failed"
                    self.log_error("execute", f"Error in checkpoint {checkpoint_name}: {str(e)}")
                    raise
                    
            self.log_checkpoint("execute", "completed", {
                "start_checkpoint": start_checkpoint,
                "end_checkpoint": end_checkpoint
            })
            
            return setup_data
            
        except Exception as e:
            self.log_error("execute", f"Setup pipeline execution failed: {str(e)}")
            self.log_checkpoint("execute", "failed", {"error": str(e)})
            raise
            
    def _get_filtered_pipeline(self, start_checkpoint: Optional[int] = None, end_checkpoint: Optional[int] = None) -> List[Dict]:
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
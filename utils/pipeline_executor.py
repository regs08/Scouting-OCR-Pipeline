from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
from .base_processor import BaseProcessor
from .path_manager import PathManager
from .directory_manager import DirectoryManager

class PipelineExecutor(BaseProcessor):
    """Manages the execution of processing pipelines."""
    
    def __init__(self,
                 path_manager: PathManager,
                 directory_manager: DirectoryManager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the pipeline executor.
        
        Args:
            path_manager: PathManager instance for handling file paths
            directory_manager: DirectoryManager instance for managing directories
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
            log_dir=log_dir or "logs/pipeline",
            operation_name=operation_name or "pipeline_executor"
        )
        
        self.path_manager = path_manager
        self.directory_manager = directory_manager
        self.session_id = session_id
        self.pipeline: List[Tuple[BaseProcessor, str, int]] = []
        self.checkpoint_status: Dict[str, str] = {}
        
    def add_processor(self, processor: BaseProcessor, checkpoint_name: str, checkpoint_num: int) -> None:
        """
        Add a processor to the pipeline.
        
        Args:
            processor: The processor to add
            checkpoint_name: Name of the checkpoint
            checkpoint_num: Number of the checkpoint
        """
        self.pipeline.append((processor, checkpoint_name, checkpoint_num))
        self.checkpoint_status[checkpoint_name] = "pending"
        self.log_info("add_processor", f"Added processor for checkpoint {checkpoint_num}: {checkpoint_name}")
        
    def _get_filtered_pipeline(self, start_checkpoint: Optional[int] = None, 
                             end_checkpoint: Optional[int] = None) -> List[Tuple[BaseProcessor, str]]:
        """
        Get the filtered pipeline based on checkpoint range.
        
        Args:
            start_checkpoint: Optional starting checkpoint number
            end_checkpoint: Optional ending checkpoint number
            
        Returns:
            List of (processor, checkpoint_name) tuples
        """
        # Sort processors by checkpoint number
        sorted_processors = sorted(self.pipeline, key=lambda x: x[2])
        
        # Filter based on checkpoint range
        if start_checkpoint is not None:
            sorted_processors = [p for p in sorted_processors if p[2] >= start_checkpoint]
        if end_checkpoint is not None:
            sorted_processors = [p for p in sorted_processors if p[2] <= end_checkpoint]
                               
        # Return just the processor and checkpoint name
        return [(p[0], p[1]) for p in sorted_processors]
        
    def execute(self, start_checkpoint: Optional[int] = None, end_checkpoint: Optional[int] = None) -> None:
        """
        Execute the processing pipeline.
        
        Args:
            start_checkpoint: Optional starting checkpoint number
            end_checkpoint: Optional ending checkpoint number
        """
        # Get filtered pipeline based on checkpoint range
        pipeline = self._get_filtered_pipeline(start_checkpoint, end_checkpoint)
        
        # Execute each processor in the pipeline
        for checkpoint_num, processor_info in enumerate(pipeline, 1):
            processor, checkpoint_name = processor_info
            try:
                # Get the checkpoint path
                checkpoint_path = self.path_manager.get_checkpoint_path(
                    session_id=self.session_id,
                    checkpoint=checkpoint_name
                )
                
                # Process the images
                processor.process(checkpoint_path)
                
                # Update checkpoint status
                self.checkpoint_status[checkpoint_name] = "completed"
                
            except Exception as e:
                self.checkpoint_status[checkpoint_name] = "failed"
                raise RuntimeError(f"Pipeline execution failed at checkpoint {checkpoint_name}: {str(e)}")
                
    def get_checkpoint_status(self, checkpoint_name: str) -> str:
        """Get the status of a specific checkpoint."""
        return self.checkpoint_status.get(checkpoint_name, "not_started")
        
    def list_checkpoints(self) -> List[Tuple[int, str]]:
        """List all available checkpoints with their numbers and names."""
        return [(num, name) for _, name, num in self.pipeline] 
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
from ..base_processor import BaseProcessor

class BaseDataProcessor(BaseProcessor):
    """Base class for data processors that handle specific data types."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the data processor.
        
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
            operation_name=operation_name
        )
        self.path_manager = path_manager
        self.session_id = session_id
        
    def process(self, input_data: Any) -> Any:
        """
        Process the input data and return the result.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
            
        Raises:
            TypeError: If input_data is not of the expected type
        """
        if not isinstance(input_data, self.get_input_type()):
            raise TypeError(f"Expected {self.get_input_type().__name__}, got {type(input_data).__name__}")
            
        return self._process_impl(input_data)
        
    def _process_impl(self, input_data: Any) -> Any:
        """
        Implementation of the processing logic.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        raise NotImplementedError("Subclasses must implement _process_impl")
        
    @classmethod
    def get_input_type(cls) -> type:
        """Get the expected input type for this processor."""
        raise NotImplementedError("Subclasses must implement get_input_type")
        
    @classmethod
    def get_output_type(cls) -> type:
        """Get the output type for this processor."""
        raise NotImplementedError("Subclasses must implement get_output_type") 
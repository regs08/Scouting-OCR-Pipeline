from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.loggable_component import BaseProcessor

class BaseSessionProcessor(BaseProcessor):
    """Base class for session processors that process data during a session."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the base session processor.
        
        Args:
            path_manager: Path manager instance
            session_id: Current session ID
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation
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
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the session data and return updated data.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Updated dictionary with processed data
        """
        raise NotImplementedError("Subclasses must implement process")
        
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the processor, implementing the RunnableComponent interface.
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing output data
        """
        # Call the process method
        result = self.process(input_data)
        
        # Handle different types of return values
        if result is None:
            # If process returns None, return the input data unchanged
            return input_data
        elif isinstance(result, dict):
            # If process returns a dict, merge it with the input data
            output_data = input_data.copy()
            output_data.update(result)
            return output_data
        else:
            # If process returns a non-dict value, wrap it in a dictionary
            output_data = input_data.copy()
            output_data['result'] = result
            return output_data 
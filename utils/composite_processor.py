import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor
from utils.pipeline_component import PipelineComponent
from utils.runnable_component import RunnableComponent

class CompositeProcessor(PipelineComponent):
    """
    A processor that can contain and execute a pipeline of sub-components.
    This allows for creating complex processors composed of simpler ones.
    """
    
    def __init__(self,
                 processor_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the composite processor.
        
        Args:
            processor_func: Optional function to process data before/after the pipeline
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
            operation_name=operation_name or "composite_processor"
        )
        
        self.processor_func = processor_func
        
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data before running the pipeline.
        If processor_func is provided, it will be called here.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed data to be passed to the pipeline
        """
        if self.processor_func:
            self.log_info("process_before_pipeline", "Executing custom processor function")
            return self.processor_func(input_data)
        return input_data
        
    @classmethod
    def create_from_components(cls, 
                             components: List[RunnableComponent],
                             processor_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                             verbose: bool = True,
                             enable_logging: bool = True,
                             enable_console: bool = True,
                             log_dir: Optional[Union[str, Path]] = None,
                             operation_name: Optional[str] = None) -> 'CompositeProcessor':
        """
        Create a composite processor with a list of components.
        
        Args:
            components: List of components to add to the pipeline
            processor_func: Optional function to process data before/after the pipeline
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
            
        Returns:
            A composite processor containing the given components
        """
        composite = cls(
            processor_func=processor_func,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name
        )
        
        # Add components to the pipeline
        for i, component in enumerate(components):
            checkpoint_number = i + 1
            component_name = component.__class__.__name__
            checkpoint_name = f"ckpt{checkpoint_number}_{component_name}"
            
            composite.add_component(
                component,
                checkpoint_name,
                checkpoint_number
            )
            
        return composite 
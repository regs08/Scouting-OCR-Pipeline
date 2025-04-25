import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor
from utils.path_manager import PathManager
from utils.runnable_component import RunnableComponent
from utils.pipeline_component import PipelineComponent
from utils.component_config import RunnableComponentConfig

class BaseManager(PipelineComponent):
    """Base class for managers that handle pipeline execution."""
    
    def __init__(self,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 component_configs: List[RunnableComponentConfig] = None,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None,
                 **kwargs: Any):
        """
        Initialize the base manager.
        
        Args:
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            component_configs: List of component configurations to initialize
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
            **kwargs: Additional keyword arguments for components, must include path_manager
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "base_manager"
        )
        
        if 'path_manager' not in kwargs:
            raise ValueError("path_manager must be provided in kwargs")
            
        self.path_manager = kwargs['path_manager']
        self.session_id = kwargs.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.pipeline = []
        self.checkpoint_status = {}
        self.component_kwargs = kwargs
        
        # Initialize components if provided
        if component_configs:
            self._init_components(component_configs)
            
    def _init_components(self, component_configs: List[RunnableComponentConfig]) -> None:
        """
        Initialize components from their configurations.
        
        Args:
            component_configs: List of component configurations
        """
        # Sort configs by checkpoint number
        sorted_configs = sorted(component_configs, key=lambda x: x.checkpoint_number)
        
        for config in sorted_configs:
            # Log component initialization
            self.log_info(
                "_init_components",
                f"Initializing {config.component_class.__name__} at checkpoint {config.checkpoint_number}"
                + (f": {config.description}" if config.description else "")
            )
            
            # Initialize the component using the stored kwargs
            try:
                # Pass through the kwargs from ApplicationManager
                init_kwargs = {
                    **self.component_kwargs,  # All kwargs from ApplicationManager
                    'component_configs': config.component_configs  # Add nested configs
                }
                
                component = config.component_class(**init_kwargs)
                self.add_component(
                    component,
                    config.checkpoint_name,
                    config.checkpoint_number
                )
            except Exception as e:
                self.log_error(
                    "_init_components",
                    f"Failed to initialize {config.component_class.__name__}: {str(e)}"
                )
                raise
        
    def add_component(self, component: RunnableComponent, checkpoint_name: str, checkpoint_number: int) -> None:
        """
        Add a component to the pipeline.
        
        Args:
            component: Component to add (either a Processor or another Manager)
            checkpoint_name: Name of the checkpoint
            checkpoint_number: Number of the checkpoint
        """
        self.pipeline.append({
            'component': component,
            'checkpoint_name': checkpoint_name,
            'checkpoint_number': checkpoint_number
        })
        self.checkpoint_status[checkpoint_name] = "pending"
        
    def add_processor(self, processor: BaseProcessor, checkpoint_name: str, checkpoint_number: int) -> None:
        """
        Add a processor to the pipeline.
        This method is maintained for backward compatibility.
        
        Args:
            processor: Processor to add
            checkpoint_name: Name of the checkpoint
            checkpoint_number: Number of the checkpoint
        """
        self.add_component(processor, checkpoint_name, checkpoint_number)
        
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
        for component_info in self.pipeline:
            checkpoint_number = component_info['checkpoint_number']
            if (start_checkpoint is None or checkpoint_number >= start_checkpoint) and \
               (end_checkpoint is None or checkpoint_number <= end_checkpoint):
                filtered.append(component_info)
                
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
        
    def run_pipeline(self, input_data: Dict[str, Any], 
                    start_checkpoint: Optional[int] = None,
                    end_checkpoint: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the pipeline from start_checkpoint to end_checkpoint.
        
        Args:
            input_data: Initial input data for the pipeline
            start_checkpoint: Optional starting checkpoint number (inclusive)
            end_checkpoint: Optional ending checkpoint number (inclusive)
            
        Returns:
            Output data from the pipeline execution
        """
        pipeline = self._get_filtered_pipeline(start_checkpoint, end_checkpoint)
        output_data = input_data.copy()  # Start with the input data
        
        self.log_info("run_pipeline", f"Starting pipeline execution with {len(pipeline)} components")
        
        # Sort the pipeline by checkpoint number to ensure proper execution order
        sorted_pipeline = sorted(pipeline, key=lambda x: x['checkpoint_number'])
        
        for component_info in sorted_pipeline:
            component = component_info['component']
            checkpoint_name = component_info['checkpoint_name']
            checkpoint_number = component_info['checkpoint_number']
            
            try:
                # Log the start of the checkpoint
                self.checkpoint_status[checkpoint_name] = "running"
                self.log_checkpoint(checkpoint_name, "started", {
                    "checkpoint_number": checkpoint_number,
                    "component_type": component.__class__.__name__
                })
                
                # Run the component
                output_data = component.run(output_data)
                
                # Log successful completion
                self.checkpoint_status[checkpoint_name] = "completed"
                self.log_checkpoint(checkpoint_name, "completed")
                
            except Exception as e:
                # Log failure
                self.checkpoint_status[checkpoint_name] = "failed"
                error_msg = f"Error in checkpoint {checkpoint_name}: {str(e)}"
                self.log_error("run_pipeline", error_msg)
                self.log_checkpoint(checkpoint_name, "failed", {"error": str(e)})
                
                # Re-raise if this is a critical error that should stop the pipeline
                raise
                
        self.log_info("run_pipeline", f"Pipeline execution completed successfully")
        return output_data
        
    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the manager's pipeline.
        
        Args:
            input_data: Optional input data. If None, will create from manager's state.
            
        Returns:
            Dict containing:
                - checkpoint_status: Status of each checkpoint
                - output_data: Final output data from pipeline
        """
        # If no input data provided, create from manager's state
        if input_data is None:
            input_data = {
                **self.component_kwargs,  # Include all kwargs passed during initialization
                'path_manager': self.path_manager,
                'session_id': self.session_id
            }
        
        try:
            # Run the pipeline
            output_data = self.run_pipeline(input_data)
            
            # Return both checkpoint status and output data
            return {
                'checkpoint_status': self.checkpoint_status,
                'output_data': output_data
            }
            
        except Exception as e:
            self.log_error("run", f"Pipeline execution failed: {str(e)}")
            return {
                'checkpoint_status': self.checkpoint_status,
                'error': str(e)
            } 
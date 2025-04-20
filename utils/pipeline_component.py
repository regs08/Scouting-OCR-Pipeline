import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor
from utils.runnable_component import RunnableComponent

class PipelineComponent(BaseProcessor):
    """
    Base class for components that can contain and execute a pipeline of other components.
    This serves as a common base for both processors and managers.
    """
    
    def __init__(self,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the pipeline component.
        
        Args:
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
            operation_name=operation_name or "pipeline_component"
        )
        
        self.pipeline = []
        self.checkpoint_status = {}
        
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
        
    def _get_filtered_pipeline(self, start_checkpoint: Optional[int] = None, 
                             end_checkpoint: Optional[int] = None) -> List[Dict]:
        """
        Get the filtered pipeline based on checkpoint range.
        
        Args:
            start_checkpoint: Starting checkpoint number (inclusive)
            end_checkpoint: Ending checkpoint number (inclusive)
            
        Returns:
            Filtered list of component information
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
        
        if not pipeline:
            self.log_info("run_pipeline", "Pipeline is empty, nothing to execute")
            return output_data
            
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
        
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data before running the pipeline.
        This can be overridden by subclasses to perform pre-pipeline processing.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed data to be passed to the pipeline
        """
        return input_data
        
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pipeline output data.
        This can be overridden by subclasses to perform post-pipeline processing.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Final processed output data
        """
        return pipeline_output
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data, optionally running a pipeline.
        This is maintained for backward compatibility with BaseProcessor.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed output data
        """
        # Process data before pipeline
        pre_data = self.process_before_pipeline(input_data)
        
        # If there's no pipeline, return the pre-processed data
        if not self.pipeline:
            return pre_data
            
        # Run the pipeline
        pipeline_output = self.run_pipeline(pre_data)
        
        # Process data after pipeline
        return self.process_after_pipeline(pipeline_output)
        
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the component, executing the process method.
        This implements the RunnableComponent interface.
        
        Args:
            input_data: Input data for the component
            
        Returns:
            Output data from the component
        """
        return self.process(input_data) 
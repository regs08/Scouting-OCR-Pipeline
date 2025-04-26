import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from datetime import datetime
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.loggable_component import LoggableComponent
from utils.runnable_component import RunnableComponent
from utils.runnable_component_config import RunnableComponentConfig

class PipelineComponent(LoggableComponent):
    """
    Base class for components that can contain and execute a pipeline of other components.
    This serves as a common base for both processors and managers, providing pipeline infrastructure
    while allowing specialized behavior at different abstraction levels.
    """
    
    
    def __init__(self,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 component_configs: Optional[List[RunnableComponentConfig]] = None,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None,
                 parent_logger: Optional[logging.Logger] = None,
                 **kwargs: Any):
        """
        Initialize the pipeline component.
        
        Args:
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            component_configs: Optional list of component configurations to initialize
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
            parent_logger: Optional parent logger for hierarchical logging
            **kwargs: Additional keyword arguments for component initialization
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or self.__class__.__name__.lower(),
            parent_logger=parent_logger
        )
        
        self.pipeline = []
        self.checkpoint_status = {}
        self.component_kwargs = {
            **kwargs,
            'parent_logger': self.logger,  # Pass this component's logger as parent to sub-components
            'log_dir': self.log_dir,  # Share the same log directory
            'enable_logging': self.enable_logging,
            'enable_console': self.enable_console,
            'verbose': self.verbose
        }
        self.session_id = kwargs.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Initialize components if provided
        if component_configs:
            self._init_components(component_configs)

    def validate_component(self, component: Any) -> bool:
        """
        Validate whether a component can be added to this pipeline.
        Simply checks if it's a PipelineComponent.
        
        Args:
            component: Component to validate
            
        Returns:
            bool: Whether the component is valid for this pipeline
        """
        if not isinstance(component, PipelineComponent):
            self.log_error("validate_component", 
                "Component must be a PipelineComponent",
                {"component": component.__class__.__name__}
            )
            return False
        return True

    def _init_components(self, component_configs: List[RunnableComponentConfig]) -> None:
        """
        Initialize components from their configurations.
        
        Args:
            component_configs: List of component configurations
        """
        sorted_configs = sorted(component_configs, key=lambda x: x.checkpoint_number)
        
        for config in sorted_configs:
            self.log_info(
                "_init_components",
                f"Initializing component at checkpoint {config.checkpoint_number}",
                {
                    "component_class": config.component_class.__name__,
                    "checkpoint_name": config.checkpoint_name,
                    "description": config.description
                }
            )
            
            try:
                init_kwargs = {
                    **self.component_kwargs,
                    'operation_name': config.checkpoint_name,
                    'component_configs': config.component_configs,
                    **config.metadata  # Pass additional metadata to component
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
                    "Failed to initialize component",
                    {
                        "component_class": config.component_class.__name__,
                        "checkpoint_number": config.checkpoint_number,
                        "error": str(e)
                    }
                )
                raise

    def add_component(self, component: RunnableComponent, checkpoint_name: str, checkpoint_number: int) -> None:
        """
        Add a component to the pipeline after validation.
        
        Args:
            component: Component to add
            checkpoint_name: Name of the checkpoint
            checkpoint_number: Number of the checkpoint
            
        Raises:
            ValueError: If component validation fails
        """
        if not self.validate_component(component):
            raise ValueError(
                f"Component {component.__class__.__name__} is not valid for {self.__class__.__name__}"
            )
            
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
            Dict containing pipeline execution results and status
        """
        pipeline = self._get_filtered_pipeline(start_checkpoint, end_checkpoint)
        output_data = input_data.copy()

        if not pipeline:
            self.log_info("run_pipeline", "Pipeline is empty, nothing to execute")
            return {'output_data': output_data, 'checkpoint_status': self.checkpoint_status}

        self.log_info("run_pipeline", "Starting pipeline execution", {
            "total_components": len(pipeline),
            "start_checkpoint": start_checkpoint,
            "end_checkpoint": end_checkpoint
        })
        
        sorted_pipeline = sorted(pipeline, key=lambda x: x['checkpoint_number'])
        
        try:
            for component_info in sorted_pipeline:
                component = component_info['component']
                checkpoint_name = component_info['checkpoint_name']
                checkpoint_number = component_info['checkpoint_number']
                
                self.checkpoint_status[checkpoint_name] = "running"
                self.log_checkpoint(checkpoint_name, "started", {
                    "checkpoint_number": checkpoint_number,
                    "component_type": component.__class__.__name__,
                    "input_data_keys": list(output_data.keys())
                })
                
                output_data = component.run(output_data)
                
                self.checkpoint_status[checkpoint_name] = "completed"
                self.log_checkpoint(checkpoint_name, "completed", {
                    "output_data_keys": list(output_data.keys())
                })
                
            self.log_info("run_pipeline", "Pipeline execution completed successfully", {
                "total_completed": len(sorted_pipeline),
                "final_output_keys": list(output_data.keys())
            })
            
            return {
                'output_data': output_data,
                'checkpoint_status': self.checkpoint_status
            }
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.log_error("run_pipeline", error_msg, {
                "failed_checkpoint": checkpoint_name,
                "checkpoint_number": checkpoint_number,
                "component_type": component.__class__.__name__
            })
            return {
                'error': error_msg,
                'checkpoint_status': self.checkpoint_status
            }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data through the pipeline.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed output data
        """
        # Process data before pipeline
        pre_data = self.process_before_pipeline(input_data)
        
        # Run the pipeline
        result = self.run_pipeline(pre_data)
        
        # Handle any pipeline errors
        if 'error' in result:
            raise RuntimeError(result['error'])
            
        # Process data after pipeline
        return self.process_after_pipeline(result['output_data'])

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data before running the setup pipeline.
        Validates input directory and initializes necessary structures.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dict with validated and prepared data for the pipeline
        """
        self.log_info("process_before_pipeline", "Preparing setup pipeline", {
            "input_data_keys": list(input_data.keys())
        })
        
        # Validate input directory exists
        if not self.input_dir.exists():
            error_msg = f"Input directory does not exist: {self.input_dir}"
            self.log_error("process_before_pipeline", error_msg)
            raise ValueError(error_msg)
        
        # Add necessary context for setup components
        pipeline_data = {
            **input_data,
            'path_manager': self.path_manager,
            'site_data': self.site_data,
            'input_dir': self.input_dir
        }
        
        return pipeline_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pipeline output data.
        Validates setup completion and prepares final output.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dict with processed output data
        """
        self.log_info("process_after_pipeline", "Processing setup pipeline output", {
            "output_keys": list(pipeline_output.keys())
        })
        
        # Validate setup completion
        if not self.path_manager.validate_setup():
            error_msg = "Setup validation failed"
            self.log_error("process_after_pipeline", error_msg)
            raise RuntimeError(error_msg)
            
        # Add setup completion status
        final_output = {
            **pipeline_output,
            'setup_completed': True,
            'session_id': self.path_manager.session_id
        }
        
        self.log_info("process_after_pipeline", "Setup completed successfully", {
            "final_output_keys": list(final_output.keys())
        })
        
        return final_output

    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the component, executing the process method.
        This implements the RunnableComponent interface.
        
        Args:
            input_data: Input data for the component. If None, uses empty dict.
            
        Returns:
            Output data from the component
        """
        return self.process(input_data or {}) 
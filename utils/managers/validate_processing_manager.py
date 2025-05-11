from pathlib import Path
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from utils.pipeline_component import PipelineComponent
from utils.runnable_component_config import RunnableComponentConfig

class ValidateProcessingManager(PipelineComponent):
    """
    High-level manager for validating processing by comparing dataframes from two separate folders.
    This manager orchestrates a pipeline of components, each responsible for a specific comparison
    between dataframes found in the two input directories.
    """
    def __init__(
        self,
        component_configs: Optional[List[RunnableComponentConfig]] = None,
        verbose: bool = True,
        enable_logging: bool = True,
        enable_console: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        operation_name: Optional[str] = None,
        parent_logger: Optional[Any] = None,
        **kwargs: Any
    ):
        """
        Initialize the ValidateProcessingManager.

        Args:
            component_configs: List of comparison component configurations.
            verbose: Whether to show detailed output.
            enable_logging: Whether to enable logging to file.
            enable_console: Whether to enable console output.
            log_dir: Directory where log files will be stored.
            operation_name: Name of the current operation/checkpoint.
            parent_logger: Optional parent logger for hierarchical logging.
            **kwargs: Additional keyword arguments for component initialization.
        """
        self.input_dir = kwargs.get('input_dir', None)
        self.site_data = kwargs.get('site_data', None)
        self.path_manager = kwargs.get('path_manager', None)
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            component_configs=component_configs,
            log_dir=log_dir,
            operation_name=operation_name or "validate_processing_manager",
            parent_logger=parent_logger,
            **kwargs
        )

    # Optionally override process_before_pipeline if you want to add manager-level context.
    # Otherwise, subcomponents will handle directory logic as needed.

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the validation processing context for all pipeline components.
        Ensures path_manager, site_data, and session_id are set and passed through.
        """
        self.path_manager = input_data.get('path_manager')
        self.site_data = input_data.get('site_data')
        self.session_id = input_data.get('session_id')
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
        session_paths = self.path_manager.get_session_paths(self.session_id)

        # Get matched files from input data or initialize empty list
        self.matched_files = input_data.get('matched_files', [])
        
        # Log matched files status
        if not self.matched_files:
            self.log_warning("process_before_pipeline", "No matched files received in input data", {
                'input_data_keys': list(input_data.keys()),
                'session_id': self.session_id
            })
        else:
            self.log_info("process_before_pipeline", "Received matched files", {
                'num_matched_files': len(self.matched_files),
                'first_file': str(self.matched_files[0].gt_path) if self.matched_files else None,
                'last_file': str(self.matched_files[-1].gt_path) if self.matched_files else None
            })
        
        # Create validation directory if it doesn't exist
        validation_dir = session_paths['processed'] / 'validation'
        validation_dir.mkdir(parents=True, exist_ok=True)

        # Pass all context through for downstream components
        pipeline_data = {
            **input_data,
            'path_manager': self.path_manager,
            'site_data': self.site_data,
            'session_id': self.session_id,
            'session_paths': session_paths,
            'matched_files': self.matched_files,
            'validation_dir': validation_dir
        }
        
        self.log_info("process_before_pipeline", "Validation processing context prepared", {
            'num_matched_files': len(self.matched_files),
            'validation_dir': str(validation_dir)
        })
        
        return pipeline_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add processing metadata and log completion.
        """
        # Log all keys in pipeline output first
        self.log_info("process_after_pipeline", "Pipeline output keys", {
            'output_keys': list(pipeline_output.keys()),
            'session_id': self.session_id
        })
        
        # Get updated matched files from pipeline output
        matched_files = pipeline_output.get('matched_files', self.matched_files)
        
        # Log detailed matched files information
        if matched_files:
            self.log_info("process_after_pipeline", "Matched files details", {
                'num_matched_files': len(matched_files),
                'type': str(type(matched_files)),
                'first_file_type': str(type(matched_files[0])),
                'first_file_attrs': dir(matched_files[0]),
                'first_file_gt_path': str(matched_files[0].gt_path),
                'first_file_pred_path': str(matched_files[0].pred_path),
                'first_file_normalized_name': matched_files[0].normalized_name
            })
        
        # Log matched files status after pipeline
        if not matched_files:
            self.log_warning("process_after_pipeline", "No matched files in pipeline output", {
                'pipeline_output_keys': list(pipeline_output.keys()),
                'session_id': self.session_id
            })
        else:
            self.log_info("process_after_pipeline", "Pipeline output contains matched files", {
                'num_matched_files': len(matched_files),
                'first_file': str(matched_files[0].gt_path) if matched_files else None,
                'last_file': str(matched_files[-1].gt_path) if matched_files else None
            })
        
        self.log_info("process_after_pipeline", "Validation processing completed", {
            'num_matched_files': len(matched_files),
            'validation_completed': True
        })
        
        return {
            **pipeline_output,
            'validation_processing_completed': True,
            'session_id': self.session_id,
            'matched_files': matched_files  # Ensure matched files are passed through
        }



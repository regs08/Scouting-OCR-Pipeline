from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from utils.pipeline_component import PipelineComponent
from utils.runnable_component_config import RunnableComponentConfig

class ModelManager(PipelineComponent):
    """
    Manages the model processing pipeline and its components.
    Handles OCR and other model-related processing steps.
    """

    def __init__(self,
                 *,
                 component_configs: Optional[List[RunnableComponentConfig]] = None,
                 path_manager: Any = None,
                 site_data: Any = None,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 **kwargs: Any):
        """
        Initialize the model manager.

        Args:
            component_configs: List of model processing components
            path_manager: PathManager instance for handling paths
            site_data: Site data configuration
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            **kwargs: Additional keyword arguments for components
        """
        self.session_id = kwargs.get('session_id')
        self.path_manager = path_manager
        self.site_data = site_data
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            component_configs=component_configs,
            **kwargs
        )

        self.log_info("__init__", "Model manager initialized", {
            "site_data": getattr(self.site_data, 'site_name', None),
            "session_id": self.session_id,
            "components": [c.checkpoint_name for c in (component_configs or [])]
        })

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the model processing context for all pipeline components.
        """
        self.path_manager = input_data.get('path_manager')
        self.site_data = input_data.get('site_data')
        self.session_id = input_data.get('session_id')
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
        session_paths = self.path_manager.get_session_paths(self.session_id)

        # Validate required directories exist
        required_dirs = ['raw', 'processed', 'flagged']
        for dir_name in required_dirs:
            if dir_name not in session_paths:
                raise ValueError(f"Required directory '{dir_name}' not found in session paths")
            if not session_paths[dir_name].exists():
                raise ValueError(f"Required directory does not exist: {session_paths[dir_name]}")

        pipeline_data = {
            **input_data,
            'path_manager': self.path_manager,
            'site_data': self.site_data,
            'session_id': self.session_id,
            'session_paths': session_paths
        }

        self.log_info("process_before_pipeline", "Model pipeline preparation complete", {
            'session_id': self.session_id,
            'directories': list(session_paths.keys())
        })

        return pipeline_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add processing metadata and log completion.
        """
        self.log_info("process_after_pipeline", "Model pipeline completed", {
            "checkpoint_status": pipeline_output.get('checkpoint_status', {})
        })
        return {
            **pipeline_output,
            'model_processing_completed': True,
            'session_id': self.session_id,
            'path_manager': self.path_manager
        } 
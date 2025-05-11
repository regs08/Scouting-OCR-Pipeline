import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from datetime import datetime
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.path_manager import PathManager
from utils.pipeline_component import PipelineComponent
from utils.runnable_component_config import RunnableComponentConfig
from utils.site_data.site_data_base import SiteDataBase, FileAnalysisResult

class SetupManager(PipelineComponent):
    """
    Manages the setup of session directories and data copying.
    Handles initialization and validation of the pipeline environment.
    """
    
    def __init__(self,
                 *,  # Force keyword arguments
                 component_configs: Optional[List[RunnableComponentConfig]] = None,
                 path_manager: Optional[PathManager] = None,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 **kwargs: Any):
        """
        Initialize the setup manager.
        
        Args:
            component_configs: List of components to initialize
            path_manager: PathManager instance for managing file paths
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            **kwargs: Additional arguments for pipeline configuration
        """
        # Create session ID if not provided
        session_id = kwargs.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Initialize pipeline component with logging
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            component_configs=component_configs,
            **kwargs
        )
        
        self.path_manager = path_manager
        
        
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare setup manager with input data from parent. Do not extract dates or validate files here.
        """
        self.input_dir = Path(input_data['input_dir'])
        self.site_data = input_data['site_data']

        # Initialize path_manager if needed (using current date or a default)
        if not self.path_manager:
            site_code = getattr(self.site_data, 'site_code', 'default')
            batch = getattr(self.site_data, 'collection_date', datetime.now().strftime("%Y%m%d"))
            self.path_manager = PathManager(
                expected_site_code=site_code,
                batch=batch
            )
            self.log_info("process_before_pipeline", "Created new path manager", {
                "site_code": site_code,
                "batch": batch
            })

        # Always ensure path_manager is in input_data
        input_data['path_manager'] = self.path_manager

        self.log_info("process_before_pipeline", "Setup manager configured", {
            "input_dir": str(self.input_dir),
            "site_data": self.site_data.__class__.__name__,
            "path_manager.batch": self.path_manager.batch
        })

        return input_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally add session info and log completion, but do not block or check for specific outputs.
        """
        self.log_info("process_after_pipeline", "Setup pipeline completed", {
            "checkpoint_status": pipeline_output.get('checkpoint_status', {})
        })
        return {
            **pipeline_output,
            'setup_completed': True,
            'session_id': self.session_id,
            'path_manager': self.path_manager
        }

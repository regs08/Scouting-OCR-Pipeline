import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from datetime import datetime
import os 
# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent
from utils.site_data.site_data_base import SiteDataBase
from utils.runnable_component_config import RunnableComponentConfig

class ApplicationManager(PipelineComponent):
    """
    High-level manager that coordinates setup and session management.
    Orchestrates the complete application pipeline from setup to processing.
    """
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 site_data: Type[SiteDataBase],
                 component_configs: Optional[List[RunnableComponentConfig]] = None,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 **kwargs: Any):
        """
        Initialize the application manager.
        
        Args:
            input_dir: Directory containing input data and ground truth
            site_data: Site data configuration class
            component_configs: List of manager components (setup, session, etc.)
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            **kwargs: Additional keyword arguments for components
        """
        self.input_dir = Path(input_dir)
        self.site_data = site_data
        
        # Create session ID
        self.session_id = kwargs.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Initialize pipeline component
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="application_manager",
            component_configs=component_configs,
            session_id=self.session_id,
            **kwargs
        )
        
        # Path manager will be set by the setup manager after it analyzes files and extracts dates
        self.path_manager = None
        
        # Log initialization
        self.log_info("__init__", "Application manager initialized", {
            "input_dir": str(self.input_dir),
            "site_data": self.site_data.site_name,
            "session_id": self.session_id
        })

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the application context for all pipeline components.
        
        Args:
            input_data: Initial input data
            
        Returns:
            Dict with application context
        """
        return {
            **input_data,
            'input_dir': str(self.input_dir),
            'site_data': self.site_data,
            'session_id': self.session_id,
            'gt_dir': os.path.join(self.input_dir, 'ground_truth')
        }

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally add session info and log completion, but do not block or check for specific outputs.
        """
        self.log_info("process_after_pipeline", "Pipeline completed", {
            "checkpoint_status": pipeline_output.get('checkpoint_status', {})
        })
        return {
            **pipeline_output,
            'session_id': self.session_id,
            'completed_at': datetime.now().isoformat()
        }
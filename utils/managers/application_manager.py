import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.managers.base_manager import BaseManager
from utils.path_manager import PathManager
from utils.managers.setup_manager import SetupManager
from utils.site_data.site_data_base import SiteDataBase
from utils.component_config import RunnableComponentConfig

class ApplicationManager(BaseManager):
    """Manages the entire application pipeline from setup to processing."""
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 site_data: Type[SiteDataBase],
                 component_configs: List[RunnableComponentConfig] = None,
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
            component_configs: List of RunnableComponentConfig objects defining the pipeline components
                           Components will be initialized and added in order of checkpoint_number
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            **kwargs: Additional keyword arguments to pass to components
        """
        self.input_dir = Path(input_dir)
        self.site_data = site_data
        
        # Initialize single PathManager instance for all components
        self.path_manager = PathManager(
            expected_site_code=self.site_data.site_code,
            batch=datetime.now().strftime("%Y%m%d")
        )
        
        # Create session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare all kwargs that will flow down to components
        component_kwargs = {
            'input_dir': str(self.input_dir),
            'site_data': site_data,
            'path_manager': self.path_manager,
            **kwargs  # Add any additional kwargs passed to ApplicationManager
        }
        
        # Initialize BaseManager with configuration
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            component_configs=component_configs,
            log_dir=log_dir,
            operation_name="application_manager",
            **component_kwargs  # Pass all kwargs down to components, including path_manager
        )
        
    def update_paths(self, new_paths: Dict[str, Path]) -> None:
        """
        Update paths in the PathManager. Changes will be reflected in all components.
        
        Args:
            new_paths: Dictionary of new paths to add/update
        """
        # Since self.path_manager is shared across all components,
        # any updates here will be visible to all components
        for key, path in new_paths.items():
            setattr(self.path_manager, key, path)
            self.log_info("update_paths", f"Updated path '{key}' to {path}")

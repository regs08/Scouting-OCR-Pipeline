import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.path_manager import PathManager
from .base_manager import BaseManager
from utils.component_config import RunnableComponentConfig
from site_data import SiteDataBase
class SetupManager(BaseManager):
    """Manages the setup of session directories and data copying."""
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 site_data: Type[SiteDataBase],
                 component_configs: List[RunnableComponentConfig]=None,
                 path_manager: PathManager = None,  # Make path_manager optional since it comes from parent
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 **kwargs: Any):
        """
        Initialize the setup manager.
        
        Args:
            input_dir: Directory containing input data and ground truth
            site_data: Site data configuration class
            component_configs: List of components to initialize
            path_manager: PathManager instance (passed from parent manager)
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            **kwargs: Additional arguments passed to parent class
        """
            
        # Create session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.site_data = site_data
        
        # Initialize BaseManager with configuration
        super().__init__(
            path_manager=path_manager,  # Pass the path_manager from parent
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="setup_manager",
            component_configs=component_configs,
            input_dir=str(input_dir),
            **kwargs
        )
        
    def setup_processing_paths(self) -> None:
        """
        Set up additional paths needed for processing.
        These changes will be reflected in the shared PathManager instance.
        """
        # Example of how SetupManager can add/modify paths
        processing_dir = Path(self.component_kwargs['input_dir']) / "processing"
        processing_dir.mkdir(exist_ok=True)
        
        # Since we're using the shared PathManager instance, these updates
        # will be visible to all components that use the same PathManager
        self.path_manager.processing_dir = processing_dir
        self.log_info("setup_processing_paths", f"Added processing directory: {processing_dir}")


# Remove SessionManager class from here 
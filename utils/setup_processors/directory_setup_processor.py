from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
from utils.managers.base_manager import BaseManager
from utils.path_manager import PathManager
from utils.component_config import RunnableComponentConfig

class DirectorySetupProcessor(BaseManager):
    """Processor for setting up session directories."""
    
    def __init__(self,
                 path_manager: PathManager,
                 session_id: str,
                 component_configs: List[RunnableComponentConfig] = None,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 **kwargs: Any):
        """
        Initialize the directory setup processor.
        
        Args:
            path_manager: PathManager instance for handling paths
            session_id: Unique identifier for the session
            component_configs: List of sub-components to initialize
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory for logs
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            component_configs=component_configs,
            log_dir=log_dir,
            operation_name="directory_setup",
            **kwargs
        )
    
    def setup_directories(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the session directory structure.
        
        Args:
            input_data: Dictionary containing setup data
            
        Returns:
            Updated dictionary with directory paths
        """
        try:
            # Get session directory from path manager
            session_paths = self.path_manager.get_session_paths(self.session_id)
            session_dir = self.path_manager.base_dir / self.session_id
            
            # Create directory structure
            (session_dir / "raw" / "original" / "_by_id").mkdir(parents=True, exist_ok=True)
            (session_dir / "ground_truth").mkdir(exist_ok=True)
            (session_dir / "logs").mkdir(exist_ok=True)
            
            self.log_info("setup_directories", f"Created directory structure in {session_dir}")
            
            # Update input data with directory paths
            input_data['session_dir'] = session_dir
            input_data['session_paths'] = session_paths
            
            return input_data
            
        except Exception as e:
            self.log_error("setup_directories", f"Error creating directory structure: {str(e)}")
            raise
            
    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the directory setup process and any sub-components.
        
        Args:
            input_data: Optional input data dictionary
            
        Returns:
            Dictionary containing setup results and component statuses
        """
        # First set up the directories
        input_data = self.setup_directories(input_data or {})
        
        # Then run any sub-components using the base manager's run
        return super().run(input_data) 
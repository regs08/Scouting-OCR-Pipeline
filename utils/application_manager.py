import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_manager import BaseManager
from utils.path_manager import PathManager
from utils.setup_manager import SetupManager
from utils.session_manager import SessionManager

class ApplicationManager(BaseManager):
    """Manages the entire application pipeline from setup to processing."""
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 expected_vineyard: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the application manager.
        
        Args:
            input_dir: Directory containing input data and ground truth
            expected_vineyard: Expected vineyard name (must match exactly)
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
        """
        # Create session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize path manager
        path_manager = PathManager(
            vineyard=expected_vineyard,
            batch=datetime.now().strftime("%Y%m%d")
        )
        
        # Initialize BaseManager
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="application_manager"
        )
        
        # Store configuration
        self.input_dir = Path(input_dir)
        self.expected_vineyard = expected_vineyard
        
        # Initialize the pipeline
        self._init_pipeline()
        
    def _init_pipeline(self) -> None:
        """Initialize the application pipeline with all required components."""
        # Create the setup manager
        setup_manager = SetupManager(
            input_dir=self.input_dir,
            expected_vineyard=self.expected_vineyard,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir
        )
        
        # Create the session manager
        session_manager = SessionManager(
            path_manager=self.path_manager,
            session_id=self.session_id,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir
        )
        
        # Add setup manager as the first pipeline component
        self.add_component(
            setup_manager,
            "ckpt1_setup",
            1
        )
        
        # Add session manager as the second pipeline component
        self.add_component(
            session_manager,
            "ckpt2_session_processing",
            2
        )
        
    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the application pipeline.
        
        Args:
            input_data: Optional input data. If None, default configuration will be used.
            
        Returns:
            Dictionary containing the application results
        """
        # Initialize default application data if none provided
        if input_data is None:
            input_data = {
                'input_dir': str(self.input_dir),
                'vineyard': self.expected_vineyard,
                'session_id': self.session_id
            }
            
        try:
            # Run the pipeline
            self.log_info("run", "Starting application pipeline")
            result = self.run_pipeline(input_data)
            self.log_info("run", "Application pipeline completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error running application pipeline: {str(e)}"
            self.log_error("run", error_msg)
            
            # Still return checkpoint statuses even on error
            return {'checkpoint_status': self.checkpoint_status}
            
    def run_application(self) -> Dict[str, Any]:
        """
        Run the entire application from setup to processing.
        
        Returns:
            Dictionary containing the results of all pipeline steps
        """
        try:
            # Run the pipeline
            return self.run()
        except Exception as e:
            self.log_error("run_application", f"Error running application: {str(e)}")
            return {'checkpoint_status': self.checkpoint_status} 
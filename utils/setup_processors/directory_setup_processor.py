from typing import Dict, Any
from pathlib import Path
from .base_setup_processor import BaseSetupProcessor

class DirectorySetupProcessor(BaseSetupProcessor):
    """Processor for setting up session directories."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
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
            
            self.log_info("process", f"Created directory structure in {session_dir}")
            
            # Update input data with directory paths
            input_data['session_dir'] = session_dir
            input_data['session_paths'] = session_paths
            
            return input_data
            
        except Exception as e:
            self.log_error("process", f"Error creating directory structure: {str(e)}")
            raise 
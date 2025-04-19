from typing import Dict, Any, List
from pathlib import Path
from .base_setup_processor import BaseSetupProcessor

class DataSetupProcessor(BaseSetupProcessor):
    """Processor for copying and validating data files to the session directory."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy and validate data files from input directory to session directory.
        
        Args:
            input_data: Dictionary containing setup data
            
        Returns:
            Updated dictionary with copied file paths
        """
        try:
            # Get input data directory
            input_data_dir = Path(input_data.get('input_dir')) / "data"
            
            # Get session directory
            session_dir = input_data['session_dir']
            raw_dir = session_dir / "raw" / "original" / "_by_id"
            
            # Process all data files
            copied_files = self._process_files(
                source_dir=input_data_dir,
                target_dir=raw_dir,
                file_type="data",
                file_pattern="*.png"  # Only process PNG files
            )
            
            # Update input data with copied files
            input_data['data_files'] = copied_files
            
            return input_data
            
        except Exception as e:
            self.log_error("process", f"Error copying data files: {str(e)}")
            raise 
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_setup_processor import BaseSetupProcessor
from ..site_data.site_data_base import SiteDataBase
from ..site_data.arget_singer_24 import ArgetSinger24SiteData
import shutil

class DataSetupProcessor(BaseSetupProcessor):
    """Processor for copying and validating data files to the session directory."""
    
    def __init__(self, site_data: SiteDataBase = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize site data configuration
        self.site_data = site_data or ArgetSinger24SiteData()
    
    def _validate_and_copy_file(self, 
                              source_file: Path, 
                              target_dir: Path, 
                              file_type: str) -> Optional[Path]:
        """
        Validate a file using site data validation and copy it to the target directory if valid.
        Invalid files are copied to the flagged directory.
        
        Args:
            source_file: Path to the source file
            target_dir: Directory to copy valid files to
            file_type: Type of file being processed (for logging)
            
        Returns:
            Path to the copied file if valid, None if invalid
        """
        try:
            # Validate using site data validation
            validation_result = self.site_data.validate_filename(source_file.name)
            if not validation_result.is_valid:
                # File is invalid, copy to flagged directory
                flagged_dir = self.path_manager.get_flagged_dir(self.session_id)
                flagged_path = flagged_dir / source_file.name
                flagged_dir.mkdir(parents=True, exist_ok=True)  # Ensure flagged directory exists
                shutil.copy2(str(source_file), str(flagged_path))
                self.log_warning("_validate_and_copy_file", 
                               f"Invalid {file_type} file copied to flagged: {source_file.name} - {validation_result.error_message}")
                return None
                
            # File is valid, copy to target directory
            target_file = target_dir / source_file.name
            if not target_file.exists():
                shutil.copy2(str(source_file), str(target_file))
                self.log_info("_validate_and_copy_file", 
                            f"Copied valid {file_type} file: {source_file.name}")
                return target_file
                
            return target_file
            
        except Exception as e:
            self.log_error("_validate_and_copy_file", 
                          f"Error processing {file_type} file {source_file.name}: {str(e)}")
            return None

    def _process_files(self, 
                      source_dir: Path, 
                      target_dir: Path, 
                      file_type: str,
                      file_pattern: str = "*") -> List[Path]:
        """
        Process all files in a directory, validating and copying them.
        
        Args:
            source_dir: Directory containing source files
            target_dir: Directory to copy valid files to
            file_type: Type of files being processed
            file_pattern: Pattern to match files (default: "*")
            
        Returns:
            List of paths to successfully copied files
        """
        if not source_dir.exists():
            self.log_warning("_process_files", f"No {file_type} directory found: {source_dir}")
            return []
            
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all matching files
        copied_files = []
        for source_file in source_dir.glob(file_pattern):
            target_file = self._validate_and_copy_file(
                source_file, 
                target_dir, 
                file_type
            )
            if target_file:
                copied_files.append(target_file)
                
        return copied_files

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy and validate data files from input directory to session directory.
        
        Args:
            input_data: Dictionary containing setup data
            
        Returns:
            Updated dictionary with copied file paths and site configuration
        """
        try:
            # Get input data directory
            input_data_dir = Path(input_data.get('input_dir')) / "data"
            
            # Get session directory
            session_dir = input_data['session_dir']
            raw_dir = session_dir / "raw" / "original" / "_by_id"
            
            # Process all supported file formats
            copied_files = []
            for file_pattern in self.site_data.get_supported_formats():
                self.log_info("process", f"Processing files matching pattern: {file_pattern}")
                files = self._process_files(
                    source_dir=input_data_dir,
                    target_dir=raw_dir,
                    file_type="data",
                    file_pattern=f"*{file_pattern}"
                )
                copied_files.extend(files)
            
            if not copied_files:
                self.log_warning("process", f"No supported files found in {input_data_dir}")
            else:
                self.log_info("process", f"Successfully copied {len(copied_files)} files")
            
            # Update input data with copied files and site configuration
            input_data['data_files'] = copied_files
            input_data['site_config'] = {
                'name': self.site_data.site_name,
                'code': self.site_data.site_code,
                'data_columns': self.site_data.data_cols,
                'index_columns': self.site_data.index_cols
            }
            
            return input_data
            
        except Exception as e:
            self.log_error("process", f"Error copying data files: {str(e)}")
            raise 
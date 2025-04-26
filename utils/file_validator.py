from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging
from utils.site_data.site_data_base import SiteDataBase, FileValidationResult

class FileValidator:
    """
    Validates files against a site data specification.
    Ensures files are named correctly and conform to the expected formats.
    """
    
    def __init__(self, site_data: SiteDataBase, logger: Optional[logging.Logger] = None):
        """
        Initialize the file validator.
        
        Args:
            site_data: SiteDataBase instance defining the expected format
            logger: Optional logger for validation messages
        """
        self.site_data = site_data
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_file(self, file_path: Path) -> FileValidationResult:
        """
        Validate a single file against the site data specification.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            FileValidationResult with validation status and details
        """
        if not file_path.exists():
            return FileValidationResult(
                is_valid=False,
                error_message=f"File does not exist: {file_path}"
            )
            
        # Use the site_data.validate_filename method to check the file
        return self.site_data.validate_filename(file_path.name)
    
    def validate_directory(self, directory_path: Path, 
                          recursive: bool = False,
                          stop_on_error: bool = False) -> Dict[str, Any]:
        """
        Validate all files in a directory against the site data specification.
        
        Args:
            directory_path: Path to the directory containing files to validate
            recursive: Whether to check subdirectories
            stop_on_error: Whether to stop on the first error
            
        Returns:
            Dictionary with validation results
        """
        if not directory_path.exists() or not directory_path.is_dir():
            return {
                'is_valid': False,
                'error_message': f"Directory does not exist: {directory_path}",
                'valid_files': [],
                'invalid_files': [],
                'total_files': 0
            }
        
        # Track validation results
        valid_files: List[Dict[str, Any]] = []
        invalid_files: List[Dict[str, Any]] = []
        total_files = 0
        
        # Get the pattern to use for globbing
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            # Skip directories
            if file_path.is_dir():
                continue
                
            total_files += 1
                
            # Validate the file
            result = self.validate_file(file_path)
            
            # Store result
            file_info = {
                'path': str(file_path),
                'name': file_path.name,
                'result': result
            }
            
            if result.is_valid:
                valid_files.append(file_info)
                self.logger.debug(f"Valid file: {file_path}")
            else:
                invalid_files.append(file_info)
                self.logger.warning(f"Invalid file: {file_path} - {result.error_message}")
                
                if stop_on_error:
                    break
        
        # Compile final result
        return {
            'is_valid': len(invalid_files) == 0,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_files': total_files,
            'valid_count': len(valid_files),
            'invalid_count': len(invalid_files)
        }
    
    def get_valid_extensions(self) -> Set[str]:
        """
        Get the set of valid file extensions from the site data.
        
        Returns:
            Set of valid file extensions
        """
        return set(self.site_data.supported_extensions)
    
    def get_file_pattern(self) -> str:
        """
        Get the filename pattern for valid files.
        
        Returns:
            Regex pattern for valid filenames
        """
        return self.site_data.get_filename_pattern() 
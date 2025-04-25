from typing import List, Dict, Optional, Pattern, Tuple
from pathlib import Path
import re
from datetime import datetime
from dataclasses import dataclass

@dataclass
class FileValidationResult:
    """Container for file validation results."""
    is_valid: bool
    error_message: Optional[str] = None
    extracted_data: Dict = None
    
    def __post_init__(self):
        if self.extracted_data is None:
            self.extracted_data = {}

class SiteDataBase:
    """
    Base class for site-specific data handling and file validation.
    Defines the interface and common functionality for site-specific operations.
    """
    
    def __init__(self, 
                 data_cols: List[str], 
                 index_cols: List[str],
                 location_pattern: str,
                 site_name: str,
                 site_code: Optional[str] = None,
                 supported_extensions: Optional[List[str]] = None):
        """
        Initialize site data configuration.
        
        Args:
            data_cols: List of column names containing actual data
            index_cols: List of column names used as indices
            location_pattern: Regex pattern for location extraction
            site_name: Name of the site
            site_code: Optional site code for file naming
            supported_extensions: List of supported file extensions
        """
        self.data_cols = data_cols
        self.index_cols = index_cols
        self.site_name = site_name
        self.site_code = site_code
        self.supported_extensions = supported_extensions or ['.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif']
        
        # Default patterns - can be overridden by child classes
        self._date_pattern = r'(\d{8})'  # YYYYMMDD
        self.location_pattern = location_pattern
        
    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        Must be implemented by child classes.
        
        Returns:
            List of column indices
        """
        raise NotImplementedError("Child classes must implement get_data_column_indices")
    
    def get_index_column_indices(self) -> List[int]:
        """
        Get the indices of columns used as indices.
        Must be implemented by child classes.
        
        Returns:
            List of column indices
        """
        raise NotImplementedError("Child classes must implement get_index_column_indices")
    
    def validate_filename(self, filename: str) -> FileValidationResult:
        """
        Validate a filename against site-specific rules.
        Can be overridden by child classes for custom validation.
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            FileValidationResult containing validation status and details
        """
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.supported_extensions:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Invalid file extension. Expected one of: {', '.join(self.supported_extensions)}"
            )
        
        # Remove extension for pattern matching
        base_name = filename[:-len(file_ext)]
        
        # Extract and validate components
        validation_result = self._validate_filename_components(base_name)
        if not validation_result.is_valid:
            return validation_result
            
        return validation_result
    
    def _validate_filename_components(self, base_name: str) -> FileValidationResult:
        """
        Validate individual components of the filename.
        Can be overridden by child classes for custom component validation.
        
        Args:
            base_name: Filename without extension
            
        Returns:
            FileValidationResult containing validation status and details
        """
        try:
            # Check site code if specified
            if self.site_code and not base_name.startswith(self.site_code):
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Invalid site code. Expected prefix: {self.site_code}"
                )
            
            # Extract date
            date_match = re.search(self._date_pattern, base_name)
            if not date_match:
                return FileValidationResult(
                    is_valid=False,
                    error_message="No valid date found. Expected format: YYYYMMDD"
                )
            
            try:
                date_str = date_match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Invalid date format: {date_str}"
                )
            
            # Extract locations
            locations = self._extract_locations(base_name)
            if not locations:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"No valid location identifiers found. Expected pattern: {self._location_pattern}"
                )
            
            # Return successful validation with extracted data
            return FileValidationResult(
                is_valid=True,
                extracted_data={
                    'date': date,
                    'date_str': date_str,
                    'locations': locations,
                    'site_code': self.site_code
                }
            )
            
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Error validating filename: {str(e)}"
            )
    
    def _extract_locations(self, text: str) -> List[Dict[str, int]]:
        """
        Extract location information from text.
        Can be overridden by child classes for custom location formats.
        
        Args:
            text: Text to extract locations from
            
        Returns:
            List of dictionaries containing location information
        """
        locations = []
        matches = re.finditer(self._location_pattern, text)
        
        for match in matches:
            locations.append({
                'row': int(match.group(1)),
                'panel': int(match.group(2))
            })
            
        return locations
    
    def get_file_pattern(self) -> str:
        """
        Get the complete filename pattern for this site.
        Can be overridden by child classes for custom patterns.
        
        Returns:
            Regex pattern string for valid filenames
        """
        site_prefix = f"{self.site_code}_" if self.site_code else ""
        return f"^{site_prefix}{self._date_pattern}.*{self._location_pattern}"
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return self.supported_extensions

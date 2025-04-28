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

@dataclass
class FileAnalysisResult:
    """Container for file analysis results."""
    is_valid: bool
    site_code: Optional[str] = None
    date: Optional[str] = None
    location_id: Optional[str] = None
    row: Optional[int] = None
    panel: Optional[int] = None
    error_message: Optional[str] = None

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
                 collection_date: str,
                 site_code: Optional[str] = None,
                 supported_extensions: Optional[List[str]] = None,
                 file_pattern: Optional[str] = None):
        """
        Initialize site data configuration.
        
        Args:
            data_cols: List of column names containing actual data
            index_cols: List of column names used as indices
            location_pattern: Regex pattern for location extraction
            site_name: Name of the site
            collection_date: Mandatory collection date in YYYYMMDD format
            site_code: Optional site code for file naming
            supported_extensions: List of supported file extensions
            file_pattern: Optional custom file pattern override
        """
        self.data_cols = data_cols
        self.index_cols = index_cols
        self.site_name = site_name
        self.site_code = site_code
        self.cols = self.index_cols + self.data_cols
        self.supported_extensions = supported_extensions or ['.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif']
        
        # Validate and store collection date
        if not re.match(r'^\d{8}$', collection_date):
            raise ValueError(f"Collection date must be in YYYYMMDD format, got: {collection_date}")
        self.collection_date = collection_date
        
        # Default patterns - can be overridden by child classes
        self._date_pattern = r'(\d{8})'  # YYYYMMDD format
        self.location_pattern = location_pattern
        
        # Set file naming pattern - can be overridden by child classes
        self._file_pattern = file_pattern or rf'^({site_code})_(\d{{8}})_(.+?).*$'
    
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
    
    def get_filename_pattern(self) -> str:
        """
        Get the regex pattern for file naming.
        Can be overridden by child classes for custom file naming conventions.
        
        Default format: {site_code}_{date}_{unique_id}
        
        Returns:
            Regex pattern for filename validation
        """
        return self._file_pattern
    
    def extract_location_info(self, location_part: str) -> Tuple[Optional[int], Optional[int], str]:
        """
        Extract row and panel information from the location part of a filename.
        
        Args:
            location_part: The part of the filename containing location information
            
        Returns:
            Tuple of (row, panel, location_id)
        """
        row = None
        panel = None
        location_id = location_part
        
        # First try the site-specific location pattern
        try:
            pattern_match = re.search(self.location_pattern, location_part)
            if pattern_match and len(pattern_match.groups()) >= 2:
                row = int(pattern_match.group(1))
                panel = int(pattern_match.group(2))
                return row, panel, location_id
        except (ValueError, IndexError):
            pass
            
        # Try common patterns if site-specific pattern didn't match
        patterns = [
            # R#P# pattern
            r'R(\d+)P(\d+)',
            # R#T# pattern
            r'R(\d+)T(\d+)',
            # letter-number format (A-1_B-2)
            r'([A-Za-z])-(\d+)_([A-Za-z])-(\d+)',
            # letter+number format (A1_B2)
            r'([A-Za-z])(\d+)_([A-Za-z])(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, location_part)
            if match:
                try:
                    # Different patterns have different group counts
                    if pattern == patterns[0] or pattern == patterns[1]:
                        # R#P# or R#T# patterns
                        row, panel = map(int, match.groups())
                        return row, panel, location_id
                    elif pattern == patterns[2] or pattern == patterns[3]:
                        # Letter-number patterns (skip the letters, use the numbers)
                        if pattern == patterns[2]:
                            # A-1_B-2 format
                            row_letter, row_num, panel_letter, panel_num = match.groups()
                        else:
                            # A1_B2 format
                            row_letter, row_num, panel_letter, panel_num = match.groups()
                        
                        row = int(row_num)
                        panel = int(panel_num)
                        return row, panel, location_id
                except (ValueError, IndexError):
                    continue
                
        return row, panel, location_id
    
    def analyze_filename(self, filename: str) -> FileAnalysisResult:
        """
        Analyzes a filename to extract site code, date, and location information.
        
        Args:
            filename: Name of the file to analyze
            
        Returns:
            FileAnalysisResult containing extracted components or error
        """
        # Remove file extension if present
        base_name = Path(filename).stem
        
        # Get the pattern for this site
        pattern = self.get_filename_pattern()
        
        # Match against pattern
        match = re.match(pattern, base_name)
        if not match:
            return FileAnalysisResult(
                is_valid=False,
                error_message=f"Invalid filename format. Expected pattern matching {pattern}"
            )
        
        # Extract groups from pattern
        groups = match.groups()
        
        # Extract site code and date
        site_code = groups[0]
        date_str = groups[1]
        
        # Check site code matches if specified
        if self.site_code and site_code != self.site_code:
            return FileAnalysisResult(
                is_valid=False,
                site_code=site_code,
                error_message=f"Site code '{site_code}' does not match expected '{self.site_code}'"
            )
        
        # Check that date has 8 digits (YYYYMMDD)
        if not re.match(r'^\d{8}$', date_str):
            return FileAnalysisResult(
                is_valid=False,
                site_code=site_code,
                error_message=f"Invalid date format: {date_str} - must be 8 digits (YYYYMMDD)"
            )
        
        # Extract location part - typically the third group, but we'll handle various pattern formats
        location_part = groups[2] if len(groups) > 2 else ""
        
        # Extract row and panel from location part
        row, panel, location_id = self.extract_location_info(location_part)
        
        # If we couldn't extract location info, consider invalid unless location is optional
        if (row is None or panel is None) and location_part:
            return FileAnalysisResult(
                is_valid=False,
                site_code=site_code,
                date=date_str,
                error_message=f"Could not extract row/panel from location: {location_part}"
            )
        
        return FileAnalysisResult(
            is_valid=True,
            site_code=site_code,
            date=date_str,
            location_id=location_id,
            row=row,
            panel=panel
        )
    
    def validate_filename(self, filename: str) -> FileValidationResult:
        """
        Validate a filename against site-specific rules.
        
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
        
        # Analyze the filename
        analysis_result = self.analyze_filename(filename)
        
        if not analysis_result.is_valid:
            return FileValidationResult(
                is_valid=False,
                error_message=analysis_result.error_message
            )
        
        # Build extracted data
        extracted_data = {
            'site_code': analysis_result.site_code,
            'date_str': analysis_result.date,
            'location_id': analysis_result.location_id,
            'row': analysis_result.row,
            'panel': analysis_result.panel
        }
        
        return FileValidationResult(
            is_valid=True,
            extracted_data=extracted_data
        )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return self.supported_extensions

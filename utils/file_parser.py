import re
import sys
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor

@dataclass
class FileInfo:
    vineyard: str
    date: datetime
    row_panel_pairs: List[Dict[str, int]]

class FileParser(BaseProcessor):
    def __init__(self, 
                 expected_vineyard: Optional[str] = None,
                 verbose: bool = False, 
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the file parser with logging capabilities.
        
        Args:
            expected_vineyard: The expected vineyard name for all files
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir or "logs/file_parsing",
            operation_name=operation_name or "file_parser"
        )
        # Pattern to match vineyard name (letters and underscores, can be single or multiple words)
        self.vineyard_pattern = r'^([a-zA-Z_]+(?:_[a-zA-Z_]+)*)_*'
        # Pattern to match date in format YYYYMMDD
        self.date_pattern = r'(\d{8})'
        # Pattern to match R and P numbers
        self.rp_pattern = r'R(\d+)P(\d+)'
        # Expected file extension
        self.expected_extension = '.png'
        # Track the vineyard name for validation
        self.expected_vineyard = expected_vineyard

    def validate_filename_format(self, filename: str) -> tuple[bool, Optional[str]]:
        """Validate if a filename matches the expected format.
        
        Args:
            filename: The filename to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if the filename is valid, False otherwise
            - error_message: Description of the error if invalid, None if valid
        """
        # Check file extension
        if not filename.endswith(self.expected_extension):
            return False, f"Invalid file extension. Expected {self.expected_extension}"
        
        # Remove extension for further validation
        base_name = filename[:-len(self.expected_extension)]
        
        # Check vineyard name
        vineyard_match = re.match(self.vineyard_pattern, base_name)
        if not vineyard_match:
            return False, "Invalid vineyard name format"
        
        # Check date format
        date_match = re.search(self.date_pattern, base_name)
        if not date_match:
            return False, "Invalid date format. Expected YYYYMMDD"
        
        # Extract the part after date for unique code validation
        date_end = date_match.end()
        unique_code = base_name[date_end:].strip('_')
        
        # Validate unique code format: R#P#_R#P#
        unique_code_pattern = r'^R\d{1,2}P\d{1,2}_R\d{1,2}P\d{1,2}$'
        if not re.match(unique_code_pattern, unique_code):
            return False, f"Invalid unique code format: {unique_code}. Expected format: R#P#_R#P# where # is a one or two-digit number"
        
        return True, None

    def parse_filename(self, filename: str, file_data: bytes = b"") -> Optional[FileInfo]:
        """Parse a filename and return structured information.
        
        Args:
            filename: The filename to parse
            file_data: The file data to be used if flagging is needed
        """
        self.log_checkpoint("parse_filename", "started", {"filename": filename})
        
        # First validate the filename format
        is_valid, error_message = self.validate_filename_format(filename)
        if not is_valid:
            self.log_error("parse_filename", error_message)
            self.log_checkpoint("parse_filename", "failed", {"error": error_message})
            
            # Flag the file in the system flagged directory
            from utils.directory_manager import DirectoryManager
            dir_manager = DirectoryManager(
                verbose=self.verbose,
                enable_logging=self.enable_logging,
                enable_console=self.enable_console,
                log_dir=self.log_dir,
                operation_name="file_parsing"
            )
            
            # Create details for the flag
            details = {
                "error": error_message,
                "filename": filename,
                "expected_format": "vineyard_date_R#P#R#P#.png",
                "validation_errors": [error_message]
            }
            
            # Flag the file
            dir_manager.flag_system_file(
                file_path=Path(filename),
                file_data=file_data,
                reason="invalid_format",
                details=details
            )
            
            return None
        
        try:
            # Remove .png extension
            base_name = filename.replace(self.expected_extension, '')
            
            # Extract vineyard name
            vineyard_match = re.match(self.vineyard_pattern, base_name)
            vineyard = vineyard_match.group(1).rstrip('_')
            
            # Validate vineyard consistency
            if self.expected_vineyard is None:
                self.expected_vineyard = vineyard
            elif vineyard != self.expected_vineyard:
                # Check if the vineyard name is a prefix of the expected vineyard
                if not self.expected_vineyard.startswith(vineyard):
                    error_msg = f"Vineyard mismatch. Expected {self.expected_vineyard}, got {vineyard}"
                    self.log_error("parse_filename", error_msg)
                    self.log_checkpoint("parse_filename", "failed", {"error": error_msg})
                    
                    # Flag the file for vineyard mismatch
                    from utils.directory_manager import DirectoryManager
                    dir_manager = DirectoryManager(
                        verbose=self.verbose,
                        enable_logging=self.enable_logging,
                        enable_console=self.enable_console,
                        log_dir=self.log_dir,
                        operation_name="file_parsing"
                    )
                    
                    details = {
                        "error": error_msg,
                        "filename": filename,
                        "expected_vineyard": self.expected_vineyard,
                        "actual_vineyard": vineyard,
                        "validation_errors": [error_msg]
                    }
                    
                    dir_manager.flag_system_file(
                        file_path=Path(filename),
                        file_data=file_data,
                        reason="parsing_error",
                        details=details
                    )
                    
                    return None
            
            # Extract date
            date_match = re.search(self.date_pattern, base_name)
            date_str = date_match.group(1)
            date = datetime.strptime(date_str, '%Y%m%d')
            
            # Extract all R and P pairs
            rp_matches = re.finditer(self.rp_pattern, base_name)
            row_panel_pairs = []
            for match in rp_matches:
                row = int(match.group(1))
                panel = int(match.group(2))
                row_panel_pairs.append({'row': row, 'panel': panel})
            
            file_info = FileInfo(
                vineyard=vineyard,
                date=date,
                row_panel_pairs=row_panel_pairs
            )
            
            self.log_debug("parse_filename", f"Successfully parsed filename: {filename}")
            self.log_checkpoint("parse_filename", "completed", {
                "vineyard": vineyard,
                "date": date_str,
                "row_panel_pairs": row_panel_pairs
            })
            return file_info
            
        except Exception as e:
            error_msg = f"Error parsing filename {filename}: {str(e)}"
            self.log_error("parse_filename", error_msg)
            self.log_checkpoint("parse_filename", "failed", {"error": error_msg})
            
            # Flag the file for parsing error
            from utils.directory_manager import DirectoryManager
            dir_manager = DirectoryManager(
                verbose=self.verbose,
                enable_logging=self.enable_logging,
                enable_console=self.enable_console,
                log_dir=self.log_dir,
                operation_name="file_parsing"
            )
            
            details = {
                "error": str(e),
                "filename": filename,
                "traceback": error_msg
            }
            
            dir_manager.flag_system_file(
                file_path=Path(filename),
                file_data=file_data,
                reason="parsing_error",
                details=details
            )
            
            return None

    def parse_images(self, images: List[Tuple[Path, bytes]]) -> Dict[str, List[Tuple[Path, bytes, FileInfo]]]:
        """Parse multiple images and organize by vineyard."""
        self.log_checkpoint("parse_images", "started", {"total_images": len(images)})
        
        vineyard_images = defaultdict(list)
        success_count = 0
        error_count = 0
        
        for image_path, image_data in images:
            # Parse the filename
            file_info = self.parse_filename(image_path.name, image_data)
            
            if file_info:
                # Group by vineyard
                vineyard_images[file_info.vineyard].append((image_path, image_data, file_info))
                success_count += 1
            else:
                self.log_warning("parse_images", f"Could not parse filename {image_path.name}")
                error_count += 1
        
        self.log_checkpoint("parse_images", "completed", {
            "total_images": len(images),
            "successfully_parsed": success_count,
            "failed_parsed": error_count,
            "vineyards_found": list(vineyard_images.keys())
        })
        
        self.log_operation_summary({
            "total_images": len(images),
            "successfully_parsed": success_count,
            "failed_parsed": error_count,
            "vineyards_found": list(vineyard_images.keys())
        })
        
        return vineyard_images

    def print_summary(self, vineyard_images: Dict[str, List[Tuple[Path, bytes, FileInfo]]]) -> None:
        """Print a summary of parsed images."""
        self.log_checkpoint("print_summary", "started")
        
        self.log_info("print_summary", "Parsed images summary:")
        for vineyard, vineyard_imgs in vineyard_images.items():
            self.log_info("print_summary", f"- {vineyard}: {len(vineyard_imgs)} images")
            # Print date and all row/panel info for each image
            for _, _, info in vineyard_imgs:
                date_str = info.date.strftime('%Y-%m-%d')
                # Join all row/panel pairs with underscores
                rp_pairs = '_'.join([f"R{rp['row']}P{rp['panel']}" for rp in info.row_panel_pairs])
                self.log_info("print_summary", f"  - {date_str}: {rp_pairs}")
        
        self.log_checkpoint("print_summary", "completed", {
            "total_vineyards": len(vineyard_images),
            "total_images": sum(len(imgs) for imgs in vineyard_images.values())
        })

# Example usage
if __name__ == "__main__":
    parser = FileParser(
        verbose=True,
        enable_logging=True,
        enable_console=False,  # Set to False to disable console output
        log_dir="logs/file_parsing",
        operation_name="test_file_parsing"
    )
    
    test_filenames = [
        "arget_singer_20240814_R9P13_R9P25.png",
        "arget_singer__20240814_R9P13_R9P25.png",  # Test with extra underscores
        "crittenden_20240814_R9P13_R9P25.png"
    ]
    
    # Create mock image data for testing
    test_images = [(Path(f), b"mock_data") for f in test_filenames]
    
    # Parse and organize images
    vineyard_images = parser.parse_images(test_images)
    
    # Print summary
    parser.print_summary(vineyard_images) 
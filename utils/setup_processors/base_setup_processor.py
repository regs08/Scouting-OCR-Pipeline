from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from utils.base_processor import BaseProcessor
from utils.file_parser import FileParser

class BaseSetupProcessor(BaseProcessor):
    """Base class for data setup processors that handles file validation and vineyard name enforcement."""
    
    def __init__(self,
                 path_manager,
                 session_id: str,
                 expected_vineyard: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the base setup processor.
        
        Args:
            path_manager: Path manager instance
            session_id: Current session ID
            expected_vineyard: Expected vineyard name for validation
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name
        )
        self.path_manager = path_manager
        self.session_id = session_id
        self.expected_vineyard = expected_vineyard
        self.file_parser = FileParser(
            expected_vineyard=expected_vineyard,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=f"{operation_name}_file_parser"
        )
        
    def _validate_and_copy_file(self, 
                              source_file: Path, 
                              target_dir: Path, 
                              file_type: str) -> Optional[Path]:
        """
        Validate a file and copy it to the target directory if valid.
        Invalid files are moved to the flagged directory.
        
        Args:
            source_file: Path to the source file
            target_dir: Directory to copy valid files to
            file_type: Type of file being processed (for logging)
            
        Returns:
            Path to the copied file if valid, None if invalid
        """
        try:
            # Parse and validate the file
            file_info = self.file_parser.parse_filename(source_file.name)
            
            if file_info is None:
                # File is invalid, move to flagged directory
                flagged_dir = self.path_manager.get_flagged_dir(self.session_id)
                flagged_path = flagged_dir / source_file.name
                shutil.move(str(source_file), str(flagged_path))
                self.log_warning("_validate_and_copy_file", 
                               f"Invalid {file_type} file moved to flagged: {source_file.name}")
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
            target_file = self._validate_and_copy_file(source_file, target_dir, file_type)
            if target_file:
                copied_files.append(target_file)
                
        return copied_files

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the setup step and return updated data.
        
        Args:
            input_data: Dictionary containing setup data
            
        Returns:
            Updated dictionary with processed data
        """
        raise NotImplementedError("Subclasses must implement process")
        
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the processor, implementing the RunnableComponent interface.
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing output data
        """
        # Call the process method
        result = self.process(input_data)
        
        # Handle different types of return values
        if result is None:
            # If process returns None, return the input data unchanged
            return input_data
        elif isinstance(result, dict):
            # If process returns a dict, merge it with the input data
            output_data = input_data.copy()
            output_data.update(result)
            return output_data
        else:
            # If process returns a non-dict value, wrap it in a dictionary
            output_data = input_data.copy()
            output_data['result'] = result
            return output_data 
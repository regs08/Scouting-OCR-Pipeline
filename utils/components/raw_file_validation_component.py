import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent
from utils.file_validator import FileValidator
from utils.site_data.site_data_base import SiteDataBase

class RawFileValidationComponent(PipelineComponent):
    """
    Pipeline component that finds and validates files based on site data specifications.
    """
    
    def __init__(self, 
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 recursive: bool = False,
                 stop_on_error: bool = False,
                 operation_name: str = "file_validation",
                 **kwargs: Any):
        """
        Initialize the file validation component.
        
        Args:
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            recursive: Whether to search directories recursively
            stop_on_error: Whether to stop on the first validation error
            operation_name: Name of the operation, defaults to "file_validation"
            **kwargs: Additional component initialization arguments
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            operation_name=operation_name,
            **kwargs
        )
        
        self.recursive = recursive
        self.stop_on_error = stop_on_error
        
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data before running the validation pipeline.
        
        Args:
            input_data: Input data containing input_dir and site_data
            
        Returns:
            Dict with validated and prepared data for the pipeline
        """
        self.log_info("process_before_pipeline", "Preparing file validation")
        
        # Extract required inputs
        if 'input_dir' not in input_data:
            raise ValueError("input_dir is required")
        
        if 'site_data' not in input_data:
            raise ValueError("site_data is required")
            
        self.input_dir = Path(input_data['input_dir'])
        self.site_data = input_data['site_data']
        
        # Validate input directory exists
        if not self.input_dir.exists():
            error_msg = f"Input directory does not exist: {self.input_dir}"
            self.log_error("process_before_pipeline", error_msg)
            raise ValueError(error_msg)
            
        # Validate site_data is a SiteDataBase
        if not isinstance(self.site_data, SiteDataBase):
            error_msg = f"site_data must be a SiteDataBase instance, got {type(self.site_data)}"
            self.log_error("process_before_pipeline", error_msg)
            raise ValueError(error_msg)
        
        # Create file validator
        self.file_validator = FileValidator(
            site_data=self.site_data,
            logger=self.logger
        )
        
        return input_data
    
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data after running the pipeline.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dict with validation results
        """
        # Validate files
        self.log_info("process_after_pipeline", f"Validating files in {self.input_dir}")
        
        validation_results = self.file_validator.validate_directory(
            directory_path=self.input_dir,
            recursive=self.recursive,
            stop_on_error=self.stop_on_error
        )
        
        # Log validation summary
        self.log_info("process_after_pipeline", "File validation complete", {
            "total_files": validation_results['total_files'],
            "valid_files": validation_results['valid_count'],
            "invalid_files": validation_results['invalid_count']
        })
        
        # Add validation results to output
        output = {
            **pipeline_output,
            'file_validation': validation_results,
            'valid_file_count': validation_results['valid_count'],
            'invalid_file_count': validation_results['invalid_count'],
            'valid_files': [Path(f['path']) for f in validation_results['valid_files']],
            'site_data': self.site_data
        }
        
        # Extract valid files for downstream processing
        valid_files = [
            Path(file_info['path']) 
            for file_info in validation_results['valid_files']
        ]
        
        output['valid_files'] = valid_files
        
        return output

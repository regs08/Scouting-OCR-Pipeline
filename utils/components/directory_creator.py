import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from utils.pipeline_component import PipelineComponent
from utils.path_manager import PathManager


class DirectoryCreator(PipelineComponent):
    """
    Creates the directory structure for the pipeline.
    Creates a directory based on site code, extracted date, and session ID.
    Also creates subdirectories for logs, processed, ground truth, and raw data.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the directory creator component.
        
        Args:
            **kwargs: Additional arguments passed to PipelineComponent
        """
        super().__init__(**kwargs)
        self.path_manager = None
        
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract required data from input_data.
        
        Args:
            input_data: Input data containing path_manager and site information
            
        Returns:
            Dict with validated input data
        """
        self.path_manager = input_data.get('path_manager')
        self.site_data = input_data.get('site_data')
        self.input_dir = input_data.get('input_dir')
        
        if not self.path_manager:
            self.log_error("process_before_pipeline", "No path manager provided in input data")
            raise ValueError("Path manager must be provided in input data")
            
        if not self.site_data:
            self.log_error("process_before_pipeline", "No site data provided in input data")
            raise ValueError("Site data must be provided in input data")
            
        return input_data
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the directory creation.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dict with directory creation results
        """
        prepared_data = self.process_before_pipeline(input_data)
        
        # Extract site code or use default
        
        # Get or create session ID (timestamp)
        session_id = self.session_id
        
        # Get or create date (could be extracted from data or current date)
        extracted_date = datetime.now().strftime("%Y%m%d")
        if hasattr(self.site_data, 'date'):
            extracted_date = self.site_data.date
        
        # Create base directory structure
        # Use the base_dir from path_manager, which already contains site_code and batch
        base_dir = self.path_manager.base_dir / session_id
        
        # Create subdirectories
        subdirs = {
            'logs': base_dir / 'logs',
            'processed': base_dir / 'processed',
            'ground_truth': base_dir / 'ground_truth',
            'raw': base_dir / 'raw',
            'flagged': base_dir / 'flagged'
        }
        
        directory_status = {}
        
        # Create directories
        try:
            # Create base directory first
            base_dir.mkdir(parents=True, exist_ok=True)
            directory_status['base'] = str(base_dir)
            self.log_info("process", f"Created base directory", {"path": str(base_dir)})
            
            for name, dir_path in subdirs.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                directory_status[name] = str(dir_path)
                self.log_info("process", f"Created directory: {name}", {"path": str(dir_path)})
                
            result = {
                **prepared_data,
                'directory_status': directory_status,
                'base_directory': str(base_dir),
                'created_directories': {
                    'base': str(base_dir),
                    **{k: str(v) for k, v in subdirs.items()}
                }
            }
            
            return self.process_after_pipeline(result)
            
        except Exception as e:
            self.log_error("process", f"Failed to create directories: {str(e)}")
            raise
    
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process after the directory creation.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dict with finalized directory information
        """
        self.log_info("process_after_pipeline", "Directory creation completed", {
            "created_directories": list(pipeline_output.get('created_directories', {}).keys())
        })
        
        # Update path manager with created directories if needed
        if self.path_manager and 'created_directories' in pipeline_output:
            dirs = pipeline_output['created_directories']
            # Update path manager with directory information (assuming path_manager has methods to update paths)
            if hasattr(self.path_manager, 'update_paths'):
                self.path_manager.update_paths(dirs)
        
        return pipeline_output 
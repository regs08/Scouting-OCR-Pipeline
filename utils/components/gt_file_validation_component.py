import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.pipeline_component import PipelineComponent
from utils.file_validator import FileValidator
from utils.site_data.gt_file_site_data import GTFileSiteData

class GTFileValidationComponent(PipelineComponent):
    """
    Pipeline component that finds and validates ground truth (GT) files based on GTFileSiteData.
    Copies valid files to the session's ground_truth directory.
    """
    def __init__(self, 
                 collection_date: Optional[str] = None,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 recursive: bool = False,
                 stop_on_error: bool = False,
                 operation_name: str = "gt_file_validation",
                 **kwargs: Any):
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            operation_name=operation_name,
            **kwargs
        )
        self.recursive = recursive
        self.stop_on_error = stop_on_error
        self.gt_site_data = None  # Will be set in process_before_pipeline
        self.file_validator = None

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Use gt_dir from input_data
        self.gt_input_dir = Path(input_data['gt_dir'])
        if not self.gt_input_dir.exists():
            raise ValueError(f"GT input directory does not exist: {self.gt_input_dir}")
        # Get GTFileSiteData from input_data['site_data']
        from utils.site_data.gt_file_site_data import GTFileSiteData
        site_data = GTFileSiteData(input_data.get('site_data'))
        self.gt_site_data = site_data
        self.file_validator = FileValidator(site_data=self.gt_site_data, logger=self.logger)
        # Prepare output directory for validated files
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        self.ground_truth_out_dir = None
        if self.path_manager and self.session_id:
            session_paths = self.path_manager.get_session_paths(self.session_id)
            self.ground_truth_out_dir = session_paths.get('ground_truth')
            if self.ground_truth_out_dir:
                self.ground_truth_out_dir.mkdir(parents=True, exist_ok=True)
        return input_data

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        self.log_info("process_after_pipeline", f"Validating GT files in {self.gt_input_dir}")
        validation_results = self.file_validator.validate_directory(
            directory_path=self.gt_input_dir,
            recursive=self.recursive,
            stop_on_error=self.stop_on_error
        )
        self.log_info("process_after_pipeline", "GT file validation complete", {
            "total_files": validation_results['total_files'],
            "valid_files": validation_results['valid_count'],
            "invalid_files": validation_results['invalid_count']
        })
        # Copy valid files to ground_truth_out_dir if available
        copied_files = []
        if self.ground_truth_out_dir:
            for file_info in validation_results['valid_files']:
                src = Path(file_info['path'])
                dest = self.ground_truth_out_dir / src.name
                try:
                    shutil.copy2(src, dest)
                    copied_files.append(dest)
                    self.log_info("process_after_pipeline", f"Copied GT file to session ground_truth dir", {
                        "source": str(src),
                        "destination": str(dest)
                    })
                except Exception as e:
                    self.log_error("process_after_pipeline", f"Failed to copy GT file", {
                        "source": str(src),
                        "destination": str(dest),
                        "error": str(e)
                    })
        else:
            self.log_warning("process_after_pipeline", "No session ground_truth directory found, skipping copy.")
        output = {
            **pipeline_output,
            'gt_file_validation': validation_results,
            'gt_valid_file_count': validation_results['valid_count'],
            'gt_invalid_file_count': validation_results['invalid_count'],
            'gt_valid_files': copied_files if copied_files else [Path(f['path']) for f in validation_results['valid_files']]
        }
        return output 
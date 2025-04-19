from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from .base_setup_processor import BaseSetupProcessor

class GroundTruthSetupProcessor(BaseSetupProcessor):
    """Processor for copying and validating ground truth files to the session directory."""
    
    def _validate_gt_filename(self, filename: str) -> bool:
        """
        TODO: Implement proper validation
        Currently assumes all files in input/gt_folder are correct.
        Future implementation should validate:
        - Vineyard name matches expected
        - Has 'gt' marker
        - Date is in YYYYMMDD format
        - R#P# pairs are properly formatted
        """
        return True
        
    def _validate_and_copy_file(self, 
                              source_file: Path, 
                              target_dir: Path, 
                              file_type: str) -> Optional[Path]:
        """
        Validate a ground truth file and copy it to the target directory if valid.
        Invalid files are moved to the flagged directory.
        
        Args:
            source_file: Path to the source file
            target_dir: Directory to copy valid files to
            file_type: Type of file being processed (for logging)
            
        Returns:
            Path to the copied file if valid, None if invalid
        """
        try:
            self.log_info("_validate_and_copy_file", f"Processing file: {source_file.name}")
            
            # First check if it's a CSV file
            if not source_file.name.endswith('.csv'):
                self.log_warning("_validate_and_copy_file", 
                               f"Non-CSV file skipped: {source_file.name}")
                return None
                
            # File is valid, copy to target directory
            target_file = target_dir / source_file.name
            if not target_file.exists():
                shutil.copy2(str(source_file), str(target_file))
                self.log_info("_validate_and_copy_file", 
                            f"Copied ground truth file: {source_file.name}")
                return target_file
            else:
                self.log_info("_validate_and_copy_file", 
                            f"File already exists in target: {source_file.name}")
                return target_file
            
        except Exception as e:
            self.log_error("_validate_and_copy_file", 
                          f"Error processing ground truth file {source_file.name}: {str(e)}")
            return None
            
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy and validate ground truth files from input directory to session directory.
        
        Args:
            input_data: Dictionary containing setup data
            
        Returns:
            Updated dictionary with copied ground truth file paths
        """
        try:
            # Get input ground truth directory
            input_gt_dir = Path(input_data.get('input_dir')) / "ground_truth"
            self.log_info("process", f"Looking for ground truth files in: {input_gt_dir}")
            
            # Get session directory
            session_dir = input_data['session_dir']
            gt_dir = session_dir / "ground_truth"
            self.log_info("process", f"Target ground truth directory: {gt_dir}")
            
            # Process all ground truth files
            copied_files = []
            if input_gt_dir.exists():
                # Ensure target directory exists
                gt_dir.mkdir(parents=True, exist_ok=True)
                
                # Process all CSV files
                csv_files = list(input_gt_dir.glob("*.csv"))
                self.log_info("process", f"Found {len(csv_files)} CSV files to process")
                
                for source_file in csv_files:
                    self.log_info("process", f"Processing file: {source_file.name}")
                    target_file = self._validate_and_copy_file(source_file, gt_dir, "ground truth")
                    if target_file:
                        copied_files.append(target_file)
                        self.log_info("process", f"Successfully processed: {source_file.name}")
            else:
                self.log_warning("process", f"No ground truth directory found: {input_gt_dir}")
            
            self.log_info("process", f"Successfully copied {len(copied_files)} ground truth files")
            
            # Update input data with copied files
            input_data['ground_truth_files'] = copied_files
            
            return input_data
            
        except Exception as e:
            self.log_error("process", f"Error copying ground truth files: {str(e)}")
            raise 
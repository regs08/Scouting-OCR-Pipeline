import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import re

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor
from utils.path_manager import PathManager
from utils.base_manager import BaseManager
from utils.pipeline_executor import PipelineExecutor
from utils.directory_manager import DirectoryManager
from utils.ocr_processor import OCRProcessor

class SessionManager(BaseManager):
    """Manages the processing of session data."""
    
    def __init__(self,
                 path_manager: PathManager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the session manager.
        
        Args:
            path_manager: Path manager for the session
            session_id: ID of the session to process
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="session_manager"
        )
        
    def add_processor(self, processor: BaseProcessor, checkpoint_name: str, checkpoint_number: int) -> None:
        """Add a processor to the pipeline."""
        super().add_processor(processor, checkpoint_name, checkpoint_number)
        
    def _get_previous_checkpoint_data(self, checkpoint_number: int) -> Dict[str, pd.DataFrame]:
        """
        Get data from the previous checkpoint.
        
        Args:
            checkpoint_number: Current checkpoint number
            
        Returns:
            Dictionary mapping filenames to their DataFrames
        """
        if checkpoint_number <= 1:
            self.log_info("_get_previous_checkpoint_data", 
                         "No previous checkpoint data needed for checkpoint 1")
            return {}
            
        # Get the checkpoints directory directly
        session_paths = self.path_manager.get_session_paths(self.session_id)
        checkpoints_dir = session_paths['checkpoints']
        
        if not checkpoints_dir.exists():
            self.log_warning("_get_previous_checkpoint_data", 
                            f"Checkpoints directory not found: {checkpoints_dir}")
            return {}
        
        # Load data from the checkpoints directory
        data: Dict[str, pd.DataFrame] = {}
        # Look for any files that contain ckpt[1-9] in their basename
        pattern = re.compile(r'ckpt[1-9]')
        checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and pattern.search(d.name)]
        
        if not checkpoint_dirs:
            self.log_warning("_get_previous_checkpoint_data", 
                            f"No checkpoint directories found in {checkpoints_dir}")
            return {}
        
        # Get the previous checkpoint directory
        prev_checkpoint = f"ckpt{checkpoint_number - 1}"
        prev_checkpoint_dir = next((d for d in checkpoint_dirs if prev_checkpoint in d.name), None)
        
        if not prev_checkpoint_dir:
            self.log_warning("_get_previous_checkpoint_data", 
                            f"Previous checkpoint directory {prev_checkpoint} not found")
            return {}
        
        # Read all CSV files in the checkpoint directory
        csv_files = list(prev_checkpoint_dir.glob("*.csv"))
        if not csv_files:
            self.log_warning("_get_previous_checkpoint_data", 
                            f"No CSV files found in {prev_checkpoint_dir}")
            return {}
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                # Store the DataFrame directly without any type conversion
                data[file_path.stem] = df
                self.log_info("_get_previous_checkpoint_data", 
                             f"Loaded data from {file_path.name}")
            except Exception as e:
                self.log_error("_get_previous_checkpoint_data", 
                             f"Error loading {file_path.name}: {str(e)}")
                continue
                
        if not data:
            self.log_warning("_get_previous_checkpoint_data", 
                            "No data successfully loaded from previous checkpoint")
        else:
            self.log_info("_get_previous_checkpoint_data", 
                         f"Successfully loaded {len(data)} files from previous checkpoint")
            
        return data
        
    def process_session(self) -> Dict[str, str]:
        """
        Process the session data.
        
        Returns:
            Dictionary mapping checkpoint names to their status
        """
        try:
            # Get session directory from path manager
            session_paths = self.path_manager.get_session_paths(self.session_id)
            session_dir = self.path_manager.base_dir / self.session_id
            
            if not session_dir.exists():
                self.log_error("process_session", f"Session directory not found: {session_dir}")
                return self.checkpoint_status
            
            # Execute each processor in the pipeline
            for processor_info in self.pipeline:
                processor = processor_info['processor']
                checkpoint_name = processor_info['checkpoint_name']
                checkpoint_number = processor_info['checkpoint_number']
                
                try:
                    # Get data from previous checkpoint
                    prev_checkpoint_data = self._get_previous_checkpoint_data(checkpoint_number)
                    
                    # Process the data
                    if isinstance(processor, OCRProcessor):
                        # OCR processor needs the session directory
                        processor.process(session_dir)
                    else:
                        # Other processors need the previous checkpoint data
                        if not prev_checkpoint_data:
                            self.log_warning("process_session", 
                                           f"No previous checkpoint data available for {checkpoint_name}")
                            self.checkpoint_status[checkpoint_name] = "skipped"
                            continue
                            
                        # Combine previous checkpoint data with session data
                        input_data = {
                            'session_id': self.session_id,
                            'prev_checkpoint_data': prev_checkpoint_data
                        }
                        processor.process(input_data)
                    
                    # Update checkpoint status
                    self.checkpoint_status[checkpoint_name] = "completed"
                    self.log_info("process_session", f"Completed checkpoint: {checkpoint_name}")
                    
                except Exception as e:
                    self.checkpoint_status[checkpoint_name] = "failed"
                    self.log_error("process_session", f"Error in checkpoint {checkpoint_name}: {str(e)}")
                    # Don't raise the exception - continue with next processor
                    continue
                
            return self.checkpoint_status
            
        except Exception as e:
            self.log_error("process_session", f"Error processing session: {str(e)}")
            return self.checkpoint_status 
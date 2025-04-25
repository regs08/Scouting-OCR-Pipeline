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
from utils.managers.base_manager import BaseManager
from utils.pipeline_executor import PipelineExecutor
from utils.directory_manager import DirectoryManager
from utils.session_processors.ocr_processor import OCRProcessor
from utils.session_processors.dimension_comparison_processor import DimensionComparisonProcessor
from utils.session_processors.confusion_matrix_processor import ConfusionMatrixSessionProcessor
from utils.runnable_component import RunnableComponent
from utils.session_processors.column_processor import ColumnProcessor
from utils.site_data.arget_singer_24 import ArgetSinger24SiteData
class ArgetSingerSessionManager(BaseManager):
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
        
        # Initialize with processors
        self._init_pipeline()
        
    def _init_pipeline(self) -> None:
        """Initialize the session pipeline with processors."""
        # Add OCR processor as the first component
        ocr_processor = OCRProcessor(
            path_manager=self.path_manager,
            session_id=self.session_id,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir,
            operation_name="ocr_processor"
        )
        
        self.add_component(
            ocr_processor,
            "ckpt1_ocr_processing",
            1
        )
        
        # Add Dimension Comparison processor as the second component
        dimension_processor = DimensionComparisonProcessor(
            path_manager=self.path_manager,
            session_id=self.session_id,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir,
            operation_name="dimension_comparison_processor"
        )
        
        self.add_component(
            dimension_processor,
            "ckpt2_dimension_comparison",
            2
        )
        
        # Add Column Processor as the third component
        column_processor = ColumnProcessor(
            path_manager=self.path_manager,
            session_id=self.session_id,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir,
            operation_name="column_processor"
        )
        
        self.add_component(
            column_processor,
            "ckpt3_column_correction",
            3
            )
        arg_singer_cols = ArgetSinger24SiteData().data_cols
        # Add Confusion Matrix processor as the fourth component
        confusion_matrix_processor = ConfusionMatrixSessionProcessor(
            path_manager=self.path_manager,
            session_id=self.session_id,
            cols_to_process=arg_singer_cols,
            source_checkpoint_name="ckpt3_column_correction",  # Use output from column processor
            case_sensitive=False,
            verbose=self.verbose,
            enable_logging=self.enable_logging,
            enable_console=self.enable_console,
            log_dir=self.log_dir,
            operation_name="confusion_matrix_processor"
        )
        
        self.add_component(
            confusion_matrix_processor,
            "ckpt4_confusion_matrix",  # Update checkpoint number
            4  # Update checkpoint number
        )
        
    def add_component(self, component: RunnableComponent, checkpoint_name: str, checkpoint_number: int) -> None:
        """Add a component to the pipeline."""
        super().add_component(component, checkpoint_name, checkpoint_number)
        
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
        
    def prepare_pipeline_data(self) -> Dict[str, Any]:
        """
        Prepare the initial data for the pipeline.
        
        Returns:
            Dictionary with initial pipeline data
        """
        # Get session directory from path manager
        session_paths = self.path_manager.get_session_paths(self.session_id)
        session_dir = self.path_manager.base_dir / self.session_id
            
        if not session_dir.exists():
            self.log_error("prepare_pipeline_data", f"Session directory not found: {session_dir}")
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        return {
            'session_id': self.session_id,
            'session_dir': session_dir,
            'session_paths': session_paths,
            'path_manager': self.path_manager
        }
        
    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the session processing pipeline.
        
        Args:
            input_data: Optional input data. If None, data will be prepared automatically.
            
        Returns:
            Dictionary with the processing results and checkpoint statuses
        """
        if input_data is None:
            input_data = self.prepare_pipeline_data()
            
        try:
            # Run the pipeline
            self.log_info("run", "Starting session processing pipeline")
            result = self.run_pipeline(input_data)
            
            # Add checkpoint statuses to the result
            result['checkpoint_status'] = self.checkpoint_status
            
            self.log_info("run", "Session processing pipeline completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error running session processing pipeline: {str(e)}"
            self.log_error("run", error_msg)
            
            # Still return checkpoint statuses even on error
            return {'checkpoint_status': self.checkpoint_status}
            
    def process_session(self) -> Dict[str, str]:
        """
        Process the session data.
        
        Returns:
            Dictionary mapping checkpoint names to their status
        """
        try:
            # Run the pipeline
            result = self.run()
            return result.get('checkpoint_status', self.checkpoint_status)
            
        except Exception as e:
            self.log_error("process_session", f"Error processing session: {str(e)}")
            return self.checkpoint_status 
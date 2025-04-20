import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from collections import defaultdict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.file_parser import FileParser, FileInfo
from utils.path_manager import PathManager
from .base_manager import BaseManager
from .setup_processors.directory_setup_processor import DirectorySetupProcessor
from .setup_processors.data_setup_processor import DataSetupProcessor
from .setup_processors.ground_truth_setup_processor import GroundTruthSetupProcessor
from .setup_processors.file_matching_processor import FileMatchingProcessor

class SetupManager(BaseManager):
    """Manages the setup of session directories and data copying."""
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 expected_vineyard: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the setup manager.
        
        Args:
            input_dir: Directory containing input data and ground truth
            expected_vineyard: Expected vineyard name (must match exactly)
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
        """
        # Validate vineyard name
        if not expected_vineyard:
            raise ValueError("Vineyard name cannot be empty")
            
        # Create session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize path manager with validated vineyard name
        path_manager = PathManager(
            vineyard=expected_vineyard,
            batch=datetime.now().strftime("%Y%m%d")
        )
        
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="setup_manager"
        )
        
        self.input_dir = Path(input_dir)
        self.expected_vineyard = expected_vineyard
        
        # Initialize the pipeline
        self._init_pipeline()
        
    def _init_pipeline(self) -> None:
        """Initialize the setup pipeline with all required components."""
        self.add_component(
            DirectorySetupProcessor(
                path_manager=self.path_manager,
                session_id=self.session_id,
                expected_vineyard=self.expected_vineyard,
                verbose=self.verbose,
                enable_logging=self.enable_logging,
                enable_console=self.enable_console,
                log_dir=self.log_dir,
                operation_name="directory_setup"
            ),
            "ckpt1_directory_setup",
            1
        )
        
        self.add_component(
            DataSetupProcessor(
                path_manager=self.path_manager,
                session_id=self.session_id,
                expected_vineyard=self.expected_vineyard,
                verbose=self.verbose,
                enable_logging=self.enable_logging,
                enable_console=self.enable_console,
                log_dir=self.log_dir,
                operation_name="data_setup"
            ),
            "ckpt2_data_setup",
            2
        )
        
        self.add_component(
            GroundTruthSetupProcessor(
                path_manager=self.path_manager,
                session_id=self.session_id,
                expected_vineyard=self.expected_vineyard,
                verbose=self.verbose,
                enable_logging=self.enable_logging,
                enable_console=self.enable_console,
                log_dir=self.log_dir,
                operation_name="ground_truth_setup"
            ),
            "ckpt3_ground_truth_setup",
            3
        )
        
        self.add_component(
            FileMatchingProcessor(
                path_manager=self.path_manager,
                session_id=self.session_id,
                expected_vineyard=self.expected_vineyard,
                verbose=self.verbose,
                enable_logging=self.enable_logging,
                enable_console=self.enable_console,
                log_dir=self.log_dir,
                operation_name="file_matching"
            ),
            "ckpt4_file_matching",
            4
        )
        
    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the setup pipeline.
        
        Args:
            input_data: Optional input data. If None, default setup data will be used.
            
        Returns:
            Dictionary containing the setup results
        """
        # Initialize default setup data if none provided
        if input_data is None:
            input_data = {
                'input_dir': str(self.input_dir),
                'session_id': self.session_id
            }
            
        try:
            # Run the pipeline
            self.log_info("run", "Starting setup pipeline")
            result = self.run_pipeline(input_data)
            self.log_info("run", "Setup pipeline completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error running setup pipeline: {str(e)}"
            self.log_error("run", error_msg)
            raise
            
    def setup_session(self) -> Tuple[Optional[Path], Optional[Path], List[Path]]:
        """
        Set up a new session with the current date.
        
        Returns:
            Tuple of (session_dir, session_logs_dir, data_files)
        """
        try:
            # Run the pipeline
            setup_data = self.run()
            
            # Extract results
            session_dir = setup_data.get('session_dir')
            session_logs_dir = session_dir / "logs" if session_dir else None
            data_files = setup_data.get('data_files', [])
            
            return session_dir, session_logs_dir, data_files
            
        except Exception as e:
            self.log_error("setup_session", f"Error setting up session: {str(e)}")
            return None, None, []

    def load_images(self) -> List[Tuple[Path, bytes]]:
        """Load all images from the input directory.
        
        Returns:
            List of tuples containing (image_path, image_data)
        """
        self.log_checkpoint("load_images", "started", {
            "input_dir": str(self.input_dir)
        })
        
        try:
            images = list(self.image_handler.process_images(self.input_dir))
            self.log_info("load_images", f"Successfully loaded {len(images)} images")
            self.log_checkpoint("load_images", "completed", {
                "total_images": len(images)
            })
            return images
        except ValueError as e:
            error_msg = f"Error loading images: {str(e)}"
            self.log_error("load_images", error_msg)
            self.log_checkpoint("load_images", "failed", {"error": error_msg})
            return []

    def setup_session_directories(self, vineyard: str, date: str) -> Tuple[Path, Path]:
        """
        Set up the session directories.
        
        Args:
            vineyard: Name of the vineyard
            date: Date string in YYYYMMDD format
            
        Returns:
            Tuple of (session_dir, session_logs_dir)
        """
        # Create a unique session ID using timestamp
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the base directory structure
        base_dir = self.vineyard / date
        session_dir = base_dir / session_id
        
        # Create the session-specific directory structure
        (session_dir / "raw" / "original" / "_by_id").mkdir(parents=True, exist_ok=True)
        (session_dir / "processed" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (session_dir / "flagged").mkdir(parents=True, exist_ok=True)
        (session_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Create the session logs directory
        session_logs_dir = session_dir / "logs"
        
        self.log_info("setup_session_directories", f"Created session directory: {session_dir}")
        self.log_info("setup_session_directories", f"Created session logs directory: {session_logs_dir}")
        
        return session_dir, session_logs_dir

    def copy_files_to_session(self, session_dir: Path, files: List[Tuple[Path, bytes, FileInfo]]) -> List[Path]:
        """
        Copy files to the session directory.
        
        Args:
            session_dir: Path to the session directory
            files: List of tuples containing (file_path, file_data, file_info)
            
        Returns:
            List of paths to the copied files
        """
        copied_files = []
        
        # Get the raw directory path
        raw_dir = session_dir / "raw" / "original" / "_by_id"
        
        for file_path, file_data, file_info in files:
            try:
                # Create the output path
                output_path = raw_dir / file_path.name
                
                # Write the file data
                with open(output_path, 'wb') as f:
                    f.write(file_data)
                    
                copied_files.append(output_path)
                self.log_info("copy_files_to_session", f"Copied {file_path.name} to {output_path}")
                
            except Exception as e:
                self.log_error("copy_files_to_session", f"Error copying {file_path.name}: {str(e)}")
                
        return copied_files

# Remove SessionManager class from here 
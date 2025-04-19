import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List
from datetime import datetime
import shutil
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.base_processor import BaseProcessor
from utils.file_parser import FileInfo

class DirectoryManager(BaseProcessor):
    """Manages directory creation and organization for the OCR pipeline."""
    
    def __init__(self,
                 verbose: bool = False,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the directory manager.
        
        Args:
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
            log_dir=log_dir or "logs/directory_management",
            operation_name=operation_name or "directory_manager"
        )
        self.project_root = Path(__file__).parent.parent
        self._setup_system_flagged_directory()

    def _setup_system_flagged_directory(self) -> None:
        """Set up the system-level flagged directory structure."""
        system_flagged_dir = self.project_root / "system_flagged"
        system_flagged_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different system-level flag reasons
        (system_flagged_dir / "invalid_format").mkdir(parents=True, exist_ok=True)
        (system_flagged_dir / "corrupted").mkdir(parents=True, exist_ok=True)
        (system_flagged_dir / "parsing_error").mkdir(parents=True, exist_ok=True)
        (system_flagged_dir / "metadata").mkdir(parents=True, exist_ok=True)

    def flag_system_file(self,
                        file_path: Path,
                        file_data: bytes,
                        reason: str,
                        details: Optional[dict] = None) -> Path:
        """Flag a file at the system level (before it reaches a session directory).
        
        Args:
            file_path: Path to the file to flag
            file_data: File data to be saved
            reason: Reason for flagging (e.g., 'invalid_format', 'corrupted', 'parsing_error')
            details: Optional dictionary containing additional details about the flag
            
        Returns:
            Path to the flagged file
            
        Raises:
            ValueError: If the reason is not a valid flag reason
        """
        self.log_checkpoint("flag_system_file", "started", {
            "file": str(file_path),
            "reason": reason,
            "details": details
        })
        
        # Validate reason
        valid_reasons = ["invalid_format", "corrupted", "parsing_error"]
        if reason not in valid_reasons:
            error_msg = f"Invalid system flag reason: {reason}. Must be one of {valid_reasons}"
            self.log_error("flag_system_file", error_msg)
            raise ValueError(error_msg)
        
        # Get system flagged directory for the specific reason
        flagged_dir = self.project_root / "system_flagged" / reason
        
        # Create timestamp for unique identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save the original file with timestamp
            dest_filename = f"{timestamp}_{file_path.name}"
            dest_path = flagged_dir / dest_filename
            with open(dest_path, 'wb') as f:
                f.write(file_data)
            
            # Save metadata if provided
            if details:
                metadata_dir = self.project_root / "system_flagged" / "metadata"
                metadata_filename = f"{timestamp}_{file_path.stem}_metadata.json"
                metadata_path = metadata_dir / metadata_filename
                
                # Add additional metadata
                full_details = {
                    "original_path": str(file_path),
                    "flagged_path": str(dest_path),
                    "timestamp": timestamp,
                    "reason": reason,
                    **details
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(full_details, f, indent=2)
            
            self.log_info("flag_system_file", f"System flagged file {file_path.name} for reason: {reason}")
            self.log_checkpoint("flag_system_file", "completed", {
                "file": str(file_path),
                "reason": reason,
                "destination": str(dest_path)
            })
            
            return dest_path
            
        except Exception as e:
            error_msg = f"Error system flagging file {file_path.name}: {str(e)}"
            self.log_error("flag_system_file", error_msg)
            self.log_checkpoint("flag_system_file", "failed", {"error": error_msg})
            raise

    def setup_session_directories(self, 
                                vineyard: str, 
                                date: datetime,
                                base_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Set up session-specific directories with safety checks.
        
        Args:
            vineyard: Name of the vineyard
            date: Date of the session
            base_dir: Optional base directory to use instead of project root
            
        Returns:
            Tuple of (session_dir, session_logs_dir)
        """
        self.log_checkpoint("setup_session_directories", "started", {
            "vineyard": vineyard,
            "date": date.strftime("%Y%m%d")
        })
        
        # Use provided base directory or project root
        base = Path(base_dir) if base_dir else self.project_root
        
        # Create system logs directory if it doesn't exist
        system_logs_dir = base / "system_logs"
        system_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory structure
        session_dir = base / vineyard / date.strftime("%Y%m%d")
        session_logs_dir = session_dir / "logs"
        
        # Track if directory existed
        directory_existed = session_dir.exists()
        
        # If session directory exists, use it
        if directory_existed:
            self.log_info("setup_session_directories", 
                         f"Using existing session directory: {session_dir}")
            
            # Ensure logs directory exists
            if not session_logs_dir.exists():
                session_logs_dir.mkdir(parents=True, exist_ok=True)
                self.log_info("setup_session_directories", 
                             f"Created logs directory in existing session: {session_logs_dir}")
        else:
            # Create new session directory
            session_dir.mkdir(parents=True, exist_ok=True)
            session_logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_info("setup_session_directories", 
                         f"Created new session directory: {session_dir}")
        
        self.log_checkpoint("setup_session_directories", "completed", {
            "session_dir": str(session_dir),
            "session_logs_dir": str(session_logs_dir),
            "directory_existed": directory_existed
        })
        
        # Add to operation summary
        self.log_operation_summary({
            "session_dir": str(session_dir),
            "session_logs_dir": str(session_logs_dir),
            "directory_existed": directory_existed,
            "vineyard": vineyard,
            "date": date.strftime("%Y%m%d")
        })
        
        return session_dir, session_logs_dir

    def create_directory_structure(self, session_dir: Path) -> None:
        """Create the directory structure for a session.
        
        Args:
            session_dir: Path to the session directory
        """
        self.log_checkpoint("create_directory_structure", "started", {
            "session_dir": str(session_dir)
        })
        
        try:
            # Create main directories
            raw_dir = session_dir / "raw"
            processed_dir = session_dir / "processed"
            logs_dir = session_dir / "logs"
            flagged_dir = session_dir / "flagged"
            
            # Create raw subdirectories
            original_dir = raw_dir / "original"
            by_id_dir = original_dir / "_by_id"
            
            # Remove old processed directories if they exist
            old_dirs = [
                processed_dir / "cleaned",
                processed_dir / "column_matched",
                processed_dir / "ocr_predictions"
            ]
            for old_dir in old_dirs:
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                    self.log_info("create_directory_structure", 
                                f"Removed old directory: {old_dir}")
            
            # Define checkpoint structure
            checkpoints = [
                "ckpt1_ocr_processed",    # After OCR processing
                "ckpt2_dim_matched",      # After dimension matching
                "ckpt3_col_value_matched", # After column and value matching
                "ckpt4_data_analysis",    # After data analysis
                "ckpt5_cleaned",          # After data cleaning
                "ckpt6_final"             # Final processed data
            ]
            
            # Create processed checkpoint directories
            checkpoint_dirs = [processed_dir / ckpt for ckpt in checkpoints]
            
            # Create flagged subdirectories
            invalid_format_dir = flagged_dir / "invalid_format"
            corrupted_dir = flagged_dir / "corrupted"
            processing_error_dir = flagged_dir / "processing_error"
            
            # Create all directories
            directories = [
                raw_dir, original_dir, by_id_dir,
                processed_dir, *checkpoint_dirs,
                logs_dir, flagged_dir,
                invalid_format_dir, corrupted_dir, processing_error_dir
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.log_debug("create_directory_structure", f"Created directory: {directory}")
            
            self.log_checkpoint("create_directory_structure", "completed", {
                "session_dir": str(session_dir),
                "directories_created": [str(d) for d in directories],
                "old_directories_removed": [str(d) for d in old_dirs if d.exists()]
            })
            
        except Exception as e:
            error_msg = f"Error creating directory structure: {str(e)}"
            self.log_error("create_directory_structure", error_msg)
            self.log_checkpoint("create_directory_structure", "failed", {"error": error_msg})
            raise ValueError(error_msg)

    def add_checkpoint(self, session_dir: Path, checkpoint_name: str) -> Path:
        """Add a new checkpoint directory to an existing session.
        
        Args:
            session_dir: Path to the session directory
            checkpoint_name: Name of the new checkpoint (e.g., 'ckpt7_something')
            
        Returns:
            Path to the new checkpoint directory
        """
        if not checkpoint_name.startswith('ckpt'):
            raise ValueError(f"Invalid checkpoint name: {checkpoint_name}. Must start with 'ckpt'")
        
        try:
            checkpoint_dir = session_dir / "processed" / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            self.log_info("add_checkpoint", f"Added new checkpoint directory: {checkpoint_dir}")
            return checkpoint_dir
            
        except Exception as e:
            error_msg = f"Error adding checkpoint {checkpoint_name}: {str(e)}"
            self.log_error("add_checkpoint", error_msg)
            raise ValueError(error_msg)

    def get_checkpoint_dir(self, session_dir: Path, checkpoint: str) -> Path:
        """Get the path to a specific checkpoint directory.
        
        Args:
            session_dir: Path to the session directory
            checkpoint: Checkpoint name (e.g., 'ckpt1_ocr_processed')
            
        Returns:
            Path to the checkpoint directory
        """
        if not checkpoint.startswith('ckpt'):
            raise ValueError(f"Invalid checkpoint name: {checkpoint}. Must start with 'ckpt'")
        
        checkpoint_dir = session_dir / "processed" / checkpoint
        if not checkpoint_dir.exists():
            # If checkpoint doesn't exist, create it
            self.log_warning("get_checkpoint_dir", 
                           f"Checkpoint directory {checkpoint} does not exist. Creating it.")
            checkpoint_dir = self.add_checkpoint(session_dir, checkpoint)
        
        return checkpoint_dir

    def move_to_checkpoint(self, 
                          session_dir: Path,
                          source_path: Path,
                          checkpoint: str,
                          new_filename: Optional[str] = None) -> Path:
        """Move a file to a specific checkpoint directory.
        
        Args:
            session_dir: Path to the session directory
            source_path: Path to the source file
            checkpoint: Checkpoint name (e.g., 'ckpt1_col_matched')
            new_filename: Optional new filename for the file
            
        Returns:
            Path to the new file location
        """
        self.log_checkpoint("move_to_checkpoint", "started", {
            "source_path": str(source_path),
            "checkpoint": checkpoint
        })
        
        try:
            # Get the checkpoint directory
            checkpoint_dir = self.get_checkpoint_dir(session_dir, checkpoint)
            
            # Determine the new filename
            if new_filename is None:
                new_filename = source_path.name
            
            # Create the destination path
            dest_path = checkpoint_dir / new_filename
            
            # Move the file
            source_path.rename(dest_path)
            
            self.log_info("move_to_checkpoint", 
                         f"Moved {source_path} to {dest_path}")
            self.log_checkpoint("move_to_checkpoint", "completed", {
                "source_path": str(source_path),
                "dest_path": str(dest_path)
            })
            
            return dest_path
            
        except Exception as e:
            error_msg = f"Error moving file to checkpoint: {str(e)}"
            self.log_error("move_to_checkpoint", error_msg)
            self.log_checkpoint("move_to_checkpoint", "failed", {"error": error_msg})
            raise ValueError(error_msg)

    def backup_existing_directory(self, directory: Path) -> Path:
        """Create a backup of an existing directory with timestamp.
        
        Args:
            directory: Directory to backup
            
        Returns:
            Path to the backup directory
        """
        if not directory.exists():
            return directory
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = directory.parent / f"{directory.name}_{timestamp}"
        
        self.log_info("backup_existing_directory", f"Creating backup of {directory} to {backup_dir}")
        shutil.copytree(directory, backup_dir)
        
        return backup_dir

    def copy_files_to_session(self, 
                            session_dir: Path,
                            files: List[Tuple[Path, bytes, FileInfo]]) -> List[Path]:
        """Copy files to the session's raw/original directory.
        
        Args:
            session_dir: Path to the session directory
            files: List of tuples containing (file_path, file_data, file_info)
            
        Returns:
            List of paths to the copied files in the session directory
            
        Raises:
            ValueError: If the session directory structure is invalid
        """
        self.log_checkpoint("copy_files_to_session", "started", {
            "session_dir": str(session_dir),
            "file_count": len(files)
        })
        
        # Verify session directory structure
        raw_dir = session_dir / "raw"
        original_dir = raw_dir / "original" / "_by_id"
        
        if not raw_dir.exists() or not original_dir.exists():
            error_msg = f"Invalid session directory structure. Missing required directories in {session_dir}"
            self.log_error("copy_files_to_session", error_msg)
            self.log_checkpoint("copy_files_to_session", "failed", {"error": error_msg})
            raise ValueError(error_msg)
        
        copied_files = []
        for file_path, file_data, file_info in files:
            try:
                # Create the destination path with full filename
                dest_path = original_dir / file_path.name
                
                # Write the file data
                with open(dest_path, 'wb') as f:
                    f.write(file_data)
                
                copied_files.append(dest_path)
                self.log_debug("copy_files_to_session", 
                             f"Copied {file_path.name} to {dest_path}")
                
            except Exception as e:
                error_msg = f"Error copying {file_path.name}: {str(e)}"
                self.log_error("copy_files_to_session", error_msg)
                continue
        
        self.log_info("copy_files_to_session", 
                     f"Copied {len(copied_files)}/{len(files)} files to session directory")
        self.log_checkpoint("copy_files_to_session", "completed", {
            "copied_files": len(copied_files),
            "total_files": len(files),
            "destination": str(original_dir)
        })
        
        return copied_files

    def get_files_by_row_panel(self, 
                             session_dir: Path,
                             row: int,
                             panel: int) -> List[Path]:
        """Get all files containing a specific row-panel pair.
        
        Args:
            session_dir: Path to the session directory
            row: Row number
            panel: Panel number
            
        Returns:
            List of paths to files containing the specified row-panel pair
        """
        self.log_checkpoint("get_files_by_row_panel", "started", {
            "row": row,
            "panel": panel
        })
        
        original_dir = session_dir / "raw" / "original" / "_by_id"
        if not original_dir.exists():
            return []
        
        # Pattern to match the row-panel pair
        pattern = f"R{row}P{panel}"
        
        # Find all files containing the row-panel pair
        matching_files = []
        for file_path in original_dir.iterdir():
            if file_path.is_file() and pattern in file_path.name:
                matching_files.append(file_path)
        
        self.log_checkpoint("get_files_by_row_panel", "completed", {
            "row": row,
            "panel": panel,
            "matches": len(matching_files)
        })
        
        return matching_files

    def flag_file(self,
                 session_dir: Path,
                 file_path: Path,
                 file_data: bytes,
                 reason: str,
                 details: Optional[dict] = None) -> Path:
        """Flag a file that cannot be processed and move it to the flagged directory.
        
        Args:
            session_dir: Path to the session directory
            file_path: Path to the file to flag
            file_data: File data to be saved
            reason: Reason for flagging (e.g., 'invalid_format', 'corrupted', 'processing_error')
            details: Optional dictionary containing additional details about the flag
            
        Returns:
            Path to the flagged file
            
        Raises:
            ValueError: If the reason is not a valid flag reason
        """
        self.log_checkpoint("flag_file", "started", {
            "file": str(file_path),
            "reason": reason,
            "details": details
        })
        
        # Validate reason
        valid_reasons = ["invalid_format", "corrupted", "processing_error"]
        if reason not in valid_reasons:
            error_msg = f"Invalid flag reason: {reason}. Must be one of {valid_reasons}"
            self.log_error("flag_file", error_msg)
            raise ValueError(error_msg)
        
        # Get flagged directory for the specific reason
        flagged_dir = session_dir / "flagged" / reason
        
        # Create metadata file name
        metadata_filename = f"{file_path.stem}_metadata.json"
        
        try:
            # Save the original file
            dest_path = flagged_dir / file_path.name
            with open(dest_path, 'wb') as f:
                f.write(file_data)
            
            # Save metadata if provided
            if details:
                metadata_path = flagged_dir / metadata_filename
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(details, f, indent=2)
            
            self.log_info("flag_file", f"Flagged file {file_path.name} for reason: {reason}")
            self.log_checkpoint("flag_file", "completed", {
                "file": str(file_path),
                "reason": reason,
                "destination": str(dest_path)
            })
            
            return dest_path
            
        except Exception as e:
            error_msg = f"Error flagging file {file_path.name}: {str(e)}"
            self.log_error("flag_file", error_msg)
            self.log_checkpoint("flag_file", "failed", {"error": error_msg})
            raise 
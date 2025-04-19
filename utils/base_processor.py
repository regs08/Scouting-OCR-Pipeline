import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path

class LogLevel(Enum):
    """Enum for different log levels."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

@dataclass
class LogMessage:
    """Data class for structured log messages."""
    level: LogLevel
    function: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

class BaseProcessor:
    def __init__(self, 
                 verbose: bool = False, 
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the base processor with logging capabilities.
        
        Args:
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
        """
        self.verbose = verbose
        self.enable_logging = enable_logging
        self.enable_console = enable_console
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.operation_name = operation_name or self.__class__.__name__
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration with both file and console handlers."""
        if self.enable_logging:
            # Create logger
            self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.operation_name}")
            self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
            
            # Create formatters
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_formatter = logging.Formatter('%(message)s')
            
            # Create and configure file handler
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"{self.operation_name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Create and configure console handler if console output is enabled
            if self.enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # Log initialization
            self.log_info("__init__", f"Initialized {self.__class__.__name__} for operation: {self.operation_name}")
            self.log_info("__init__", f"Log file: {log_file}")

    def _format_message(self, log_message: LogMessage) -> str:
        """Format a log message in a human-readable way."""
        # Format the timestamp
        timestamp = log_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format the main message
        formatted_msg = f"[{timestamp}] {log_message.function}: {log_message.message}"
        
        # Add data if present
        if log_message.data:
            formatted_msg += "\nAdditional Data:"
            for key, value in log_message.data.items():
                formatted_msg += f"\n  {key}: {value}"
        
        return formatted_msg

    def log(self, 
            level: LogLevel, 
            function: str, 
            message: str, 
            data: Optional[Dict[str, Any]] = None) -> None:
        """Log a message with standardized formatting."""
        log_message = LogMessage(level=level, function=function, message=message, data=data)
        formatted_msg = self._format_message(log_message)
        
        # Map our LogLevel to Python's logging levels
        log_level = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        # Log the message
        if self.enable_logging:
            self.logger.log(log_level, formatted_msg)
        elif self.enable_console and (self.verbose or level in [LogLevel.ERROR, LogLevel.WARNING, LogLevel.CRITICAL]):
            print(formatted_msg)

    def log_debug(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self.log(LogLevel.DEBUG, function, message, data)

    def log_info(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        self.log(LogLevel.INFO, function, message, data)

    def log_warning(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self.log(LogLevel.WARNING, function, message, data)

    def log_error(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        self.log(LogLevel.ERROR, function, message, data)

    def log_critical(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a critical message."""
        self.log(LogLevel.CRITICAL, function, message, data)

    def log_checkpoint(self, checkpoint_name: str, status: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a checkpoint in the processing pipeline.
        
        Args:
            checkpoint_name: Name of the checkpoint
            status: Status of the checkpoint (e.g., "started", "completed", "failed")
            data: Optional additional data about the checkpoint
        """
        message = f"Checkpoint '{checkpoint_name}' {status}"
        self.log_info("checkpoint", message, data)

    def log_operation_summary(self, operation_data: Dict[str, Any]) -> None:
        """Log a summary of the entire operation.
        
        Args:
            operation_data: Dictionary containing summary information
        """
        self.log_info("summary", f"Operation '{self.operation_name}' completed", operation_data)

# Example usage
if __name__ == "__main__":
    class ExampleProcessor(BaseProcessor):
        def process_data(self):
            # Log start of processing
            self.log_checkpoint("data_processing", "started")
            
            try:
                # Simulate some processing steps
                self.log_checkpoint("step1", "started", {"items": 100})
                self.log_debug("process_data", "Processing step 1")
                self.log_checkpoint("step1", "completed", {"processed": 100})
                
                self.log_checkpoint("step2", "started")
                self.log_debug("process_data", "Processing step 2", {"items_processed": 100})
                self.log_checkpoint("step2", "completed")
                
                # Simulate an error
                raise ValueError("Test error")
            except Exception as e:
                self.log_checkpoint("data_processing", "failed", {"error": str(e)})
                self.log_error("process_data", f"Error during processing: {str(e)}")
                return
            
            # Log successful completion
            self.log_checkpoint("data_processing", "completed")
            
            # Log operation summary
            self.log_operation_summary({
                "total_items": 100,
                "successful": 95,
                "failed": 5,
                "duration_seconds": 10.5
            })

    # Test the processor with different console/logging settings
    print("\nTest with both console and logging:")
    processor1 = ExampleProcessor(
        verbose=True,
        enable_logging=True,
        enable_console=True,
        log_dir="logs/operations",
        operation_name="test_processing"
    )
    processor1.process_data()

    print("\nTest with logging only (no console):")
    processor2 = ExampleProcessor(
        verbose=True,
        enable_logging=True,
        enable_console=False,
        log_dir="logs/operations",
        operation_name="test_processing_no_console"
    )
    processor2.process_data()

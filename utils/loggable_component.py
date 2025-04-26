import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path

from utils.runnable_component import RunnableComponent

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
    component_name: str
    function: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

class LoggableComponent(RunnableComponent):
    """
    Base class providing logging capabilities for components in the pipeline.
    Supports both component-specific logging and pipeline-wide logging through
    a parent logger if provided.
    """
    
    # Class-level pipeline logger for shared logging
    pipeline_logger: Optional[logging.Logger] = None
    
    @classmethod
    def setup_pipeline_logger(cls, 
                            log_dir: Union[str, Path],
                            pipeline_name: str = "pipeline",
                            log_level: int = logging.INFO) -> None:
        """
        Set up a shared pipeline-wide logger.
        
        Args:
            log_dir: Directory for log files
            pipeline_name: Name of the pipeline for logging
            log_level: Logging level for the pipeline logger
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline-wide logger
        cls.pipeline_logger = logging.getLogger(f"pipeline.{pipeline_name}")
        cls.pipeline_logger.setLevel(log_level)
        
        # Set up pipeline-wide file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pipeline_log_file = log_dir / f"{pipeline_name}_{timestamp}.log"
        
        handler = logging.FileHandler(pipeline_log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        cls.pipeline_logger.addHandler(handler)

    def __init__(self, 
                 verbose: bool = False, 
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None,
                 parent_logger: Optional[logging.Logger] = None,
                 **kwargs):  # Accept and ignore extra kwargs
        """
        Initialize the loggable component with logging capabilities.
        
        Args:
            verbose: Whether to show detailed debug-level output
            enable_logging: Whether to enable component-specific logging
            enable_console: Whether to enable console output
            log_dir: Directory where component log files will be stored
            operation_name: Name of the current operation for logging purposes
            parent_logger: Optional parent logger for hierarchical logging
            **kwargs: Additional arguments (ignored)
        """
        self.verbose = verbose
        self.enable_logging = enable_logging
        self.enable_console = enable_console
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.operation_name = operation_name or self.__class__.__name__
        self.parent_logger = parent_logger
        
        # Initialize component-specific logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure and set up the logging system with file and console handlers."""
        if not self.enable_logging and not self.enable_console:
            return
            
        # Create logger with component-specific name
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.operation_name}")
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        if self.enable_logging:
            self._setup_file_logging()
            
        if self.enable_console:
            self._setup_console_logging()
            
        # Log initialization
        self.log_info("__init__", f"Initialized {self.__class__.__name__} for operation: {self.operation_name}")

    def _setup_file_logging(self) -> None:
        """Set up file-based logging."""
        # Create formatters and handlers for file logging
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{self.operation_name}_{timestamp}.log"
        
        # Set up file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_info("__init__", f"Log file created at: {log_file}")

    def _setup_console_logging(self) -> None:
        """Set up console-based logging."""
        console_formatter = logging.Formatter('%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _format_message(self, log_message: LogMessage) -> str:
        """Format a log message with component context."""
        timestamp = log_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        formatted_msg = (
            f"[{timestamp}] {log_message.component_name}.{log_message.function}: "
            f"{log_message.message}"
        )
        
        if log_message.data:
            formatted_msg += "\nContext Data:"
            for key, value in log_message.data.items():
                formatted_msg += f"\n  {key}: {value}"
        
        return formatted_msg

    def log(self, 
            level: LogLevel, 
            function: str, 
            message: str, 
            data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message both to component logger and pipeline logger if available.
        
        Args:
            level: Severity level of the log message
            function: Name of the function/operation being logged
            message: Main log message
            data: Optional structured data to include in the log
        """
        if not self.enable_logging and not self.enable_console:
            return
            
        log_message = LogMessage(
            level=level,
            component_name=self.operation_name,
            function=function,
            message=message,
            data=data
        )
        
        formatted_msg = self._format_message(log_message)
        log_level = getattr(logging, level.name)
        
        # Log to component's logger
        if self.logger:
            self.logger.log(log_level, formatted_msg)
        
        # Log to pipeline logger if available
        if self.pipeline_logger:
            self.pipeline_logger.log(log_level, formatted_msg)
            
        # Log to parent logger if provided
        if self.parent_logger:
            self.parent_logger.log(log_level, formatted_msg)

    # Convenience logging methods
    def log_debug(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug-level message."""
        self.log(LogLevel.DEBUG, function, message, data)

    def log_info(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an info-level message."""
        self.log(LogLevel.INFO, function, message, data)

    def log_warning(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning-level message."""
        self.log(LogLevel.WARNING, function, message, data)

    def log_error(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an error-level message."""
        self.log(LogLevel.ERROR, function, message, data)

    def log_critical(self, function: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a critical-level message."""
        self.log(LogLevel.CRITICAL, function, message, data)

    def log_checkpoint(self, checkpoint_name: str, status: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a pipeline checkpoint status to both component and pipeline loggers.
        
        Args:
            checkpoint_name: Name of the checkpoint
            status: Status of the checkpoint (e.g., "started", "completed", "failed")
            data: Optional additional data about the checkpoint
        """
        checkpoint_data = {
            "checkpoint": checkpoint_name,
            "status": status,
            "component": self.operation_name,
            **(data or {})
        }
        
        self.log_info("checkpoint", f"Checkpoint '{checkpoint_name}' {status}", checkpoint_data)

    def log_operation_summary(self, operation_data: Dict[str, Any]) -> None:
        """
        Log a summary of the entire operation.
        
        Args:
            operation_data: Dictionary containing summary information
        """
        self.log_info("summary", f"Operation '{self.operation_name}' completed", operation_data)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by components.
        This base class focuses on logging; actual processing should be implemented by subclasses.
        
        Args:
            input_data: Dictionary containing input data for the component
            
        Returns:
            Dictionary containing output data from the component
        """
        raise NotImplementedError("Components must implement the run method")

# Example usage
if __name__ == "__main__":
    class ExampleProcessor(LoggableComponent):
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

import os
import sys
from pathlib import Path
from typing import Union, Optional, List, Generator
import imghdr
from PIL import Image

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.loggable_component import BaseProcessor

class ImageHandler(BaseProcessor):
    def __init__(self, 
                 verbose: bool = False, 
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the image handler with logging capabilities.
        
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
            log_dir=log_dir or "logs/image_processing",
            operation_name=operation_name or "image_handler"
        )
        # Common image formats supported by Azure Form Recognizer
        self.supported_formats = {
            'jpeg', 'jpg', 'png', 'bmp', 'tiff', 'heic', 'heif', 'pdf'
        }

    def validate_image(self, image_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
        """Validate if the image file exists and is in a supported format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if the image is valid, False otherwise
            - error_message: Description of the error if invalid, None if valid
        """
        try:
            # Convert to Path object
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                return False, f"Image file not found: {image_path}"
            
            # Check if it's a file (not a directory)
            if not image_path.is_file():
                return False, f"Path is not a file: {image_path}"
            
            # Check file size (Azure Form Recognizer has a 50MB limit)
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                return False, f"File size ({file_size_mb:.2f}MB) exceeds 50MB limit"
            
            # Check if file extension is supported
            if image_path.suffix.lower()[1:] not in self.supported_formats:
                return False, f"Unsupported image format: {image_path.suffix}. Supported formats: {', '.join(self.supported_formats)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"

    def load_image(self, image_path: Union[str, Path]) -> bytes:
        """Load and validate a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image data as bytes
            
        Raises:
            ValueError: If the image is invalid or cannot be loaded
        """
        self.log_checkpoint("load_image", "started", {"image_path": str(image_path)})
        
        # Validate the image first
        is_valid, error_message = self.validate_image(image_path)
        if not is_valid:
            self.log_error("load_image", error_message)
            self.log_checkpoint("load_image", "failed", {"error": error_message})
            raise ValueError(error_message)
        
        try:
            # Read the image file
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Verify the data is not empty
            if not image_data:
                error_msg = f"Image file is empty: {image_path}"
                self.log_error("load_image", error_msg)
                self.log_checkpoint("load_image", "failed", {"error": error_msg})
                raise ValueError(error_msg)
            
            self.log_debug("load_image", f"Successfully loaded image: {image_path}")
            self.log_checkpoint("load_image", "completed", {
                "file_size": len(image_data),
                "file_path": str(image_path)
            })
            return image_data
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            self.log_error("load_image", error_msg)
            self.log_checkpoint("load_image", "failed", {"error": error_msg})
            raise ValueError(error_msg)


    def get_image_files(self, path: Union[str, Path]) -> List[Path]:
        """Get all image files from a directory or return a single image path.
        
        Args:
            path: Path to a directory or single image file
            
        Returns:
            List of Path objects for image files
            
        Raises:
            ValueError: If the path doesn't exist or contains no valid images
        """
        self.log_checkpoint("get_files", "started", {"path": str(path)})
        
        path = Path(path)
        if not path.exists():
            error_msg = f"Path does not exist: {path}"
            self.log_error("get_image_files", error_msg)
            self.log_checkpoint("get_files", "failed", {"error": error_msg})
            raise ValueError(error_msg)
        
        if path.is_file():
            if path.suffix.lower()[1:] in self.supported_formats:
                self.log_info("get_image_files", f"Found single image file: {path}")
                self.log_checkpoint("get_files", "completed", {
                    "file_count": 1,
                    "files": [str(path)]
                })
                return [path]
            error_msg = f"File is not a supported image format: {path}"
            self.log_error("get_image_files", error_msg)
            self.log_checkpoint("get_files", "failed", {"error": error_msg})
            raise ValueError(error_msg)
        
        # Get all image files from directory
        image_files = []
        for file in path.iterdir():
            if file.is_file() and file.suffix.lower()[1:] in self.supported_formats:
                image_files.append(file)
        
        if not image_files:
            error_msg = f"No supported image files found in directory: {path}"
            self.log_error("get_image_files", error_msg)
            self.log_checkpoint("get_files", "failed", {"error": error_msg})
            raise ValueError(error_msg)
        
        self.log_info("get_image_files", 
                     f"Found {len(image_files)} image files in directory",
                     {"directory": str(path), "files": [str(f) for f in image_files]})
        self.log_checkpoint("get_files", "completed", {
            "file_count": len(image_files),
            "files": [str(f) for f in image_files]
        })
        return image_files

    def process_images(self, path: Union[str, Path]) -> Generator[tuple[Path, bytes], None, None]:
        """Process all images in a directory or a single image.
        
        Args:
            path: Path to a directory or single image file
            
        Yields:
            Tuples of (image_path, image_data) for each valid image
            
        Raises:
            ValueError: If the path doesn't exist or contains no valid images
        """
        self.log_checkpoint("process_images", "started", {"path": str(path)})
        
        image_files = self.get_image_files(path)
        processed_count = 0
        error_count = 0
        
        for image_path in image_files:
            try:
                image_data = self.load_image(image_path)
                self.log_debug("process_images", f"Successfully processed image: {image_path.name}")
                processed_count += 1
                yield image_path, image_data
            except ValueError as e:
                self.log_warning("process_images", f"Skipping {image_path.name}", {"error": str(e)})
                error_count += 1
                continue
        
        self.log_checkpoint("process_images", "completed", {
            "total_images": len(image_files),
            "processed_successfully": processed_count,
            "errors": error_count
        })
        
        self.log_operation_summary({
            "total_images": len(image_files),
            "processed_successfully": processed_count,
            "errors": error_count,
            "path": str(path)
        })

# Example usage
if __name__ == "__main__":
    handler = ImageHandler(
        verbose=True,
        enable_logging=True,
        enable_console=True,  # Set to False to disable console output
        log_dir="logs/image_processing",
        operation_name="test_image_processing"
    )
    
    # Test with single image
    test_image = "test.png"
    print("\nTesting single image:")
    try:
        image_files = handler.get_image_files(test_image)
        for image_path, image_data in handler.process_images(test_image):
            width, height = handler.get_image_dimensions(image_path)
    except ValueError as e:
        print(f"Error: {str(e)}")
    
    # Test with directory
    test_dir = "test_images"
    print("\nTesting directory:")
    try:
        image_files = handler.get_image_files(test_dir)
        for image_path, image_data in handler.process_images(test_dir):
            width, height = handler.get_image_dimensions(image_path)
    except ValueError as e:
        print(f"Error: {str(e)}") 
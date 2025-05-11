#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("file_validator_test")

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.file_validator import FileValidator
from utils.site_data.arget_singer_24 import ArgetSinger24SiteData


def create_test_files(directory: Path) -> None:
    """
    Create a set of test files for validation.
    
    Args:
        directory: Directory to create files in
    """
    # Create both valid and invalid files
    
    # Valid files
    valid_files = [
        "DML_20242408_R1T2.jpg",
        "DML_20242408_R5T10.jpg",
        "DML_20242408_R1T2_extra_info.jpg",
        "AS_20242408_R10P14_R10P22.jpg",  # New format test
        "AS_20242408_R5P3_R5P4.jpg",      # New format test
        "AS_20242408_R1P1_R1P2.jpg"       # New format test
    ]
    
    # Invalid files - wrong format
    invalid_files = [
        "DMX_20242408_R1T2.jpg",   # Wrong site code
        "DML_2024240_R1T2.jpg",    # Invalid date format
        "DML_20242408_P1T2.jpg",   # Wrong location format
        "DML_20242408.jpg",        # Missing location
        "DML_20242408_R1.jpg",     # Incomplete location
        "DML_20242408_T2.jpg",     # Incomplete location
        "wrong_format.jpg",        # Completely wrong format
        "DML_20242408_R1T2.txt",   # Unsupported extension
        "AS_20242408_R1P1.jpg",    # Incomplete R##P## format
        "AS_20242408_R1P1_R1.jpg", # Incomplete second R##P## format
        "AS_20242408_R1_R1P1.jpg"  # Incomplete first R##P## format
    ]
    
    # Create directory
    directory.mkdir(parents=True, exist_ok=True)
    
    # Create valid files
    for filename in valid_files:
        file_path = directory / filename
        with open(file_path, "w") as f:
            f.write(f"Test file: {filename}")
    
    # Create invalid files
    for filename in invalid_files:
        file_path = directory / filename
        with open(file_path, "w") as f:
            f.write(f"Test file: {filename}")
    
    # Create a subdirectory with additional files
    subdir = directory / "subdir"
    subdir.mkdir(exist_ok=True)
    
    with open(subdir / "DML_20242408_R3T4.jpg", "w") as f:
        f.write("Test file in subdirectory")


def main():
    """Run the file validator tests."""
    # Set up test directory
    test_dir = Path("./test_files_validation")
    
    # Clean up any existing directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create test files
    create_test_files(test_dir)
    
    # Create site data and validator
    site_data = ArgetSinger24SiteData(collection_date="20242408")
    validator = FileValidator(site_data, logger=logger)
    
    # Print file pattern information
    print("\nFile validation settings:")
    print(f"Site code: {site_data.site_code}")
    print(f"Collection date: {site_data.collection_date}")
    print(f"File pattern: {validator.get_file_pattern()}")
    print(f"Supported extensions: {', '.join(validator.get_valid_extensions())}")
    
    # Validate a single file
    print("\nValidating a single valid file:")
    valid_file = test_dir / "DML_20242408_R1T2.jpg"
    result = validator.validate_file(valid_file)
    print(f"File: {valid_file.name}")
    print(f"Valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_message}")
    else:
        print(f"Extracted data: {result.extracted_data}")
    
    # Validate a single invalid file
    print("\nValidating a single invalid file:")
    invalid_file = test_dir / "DMX_20242408_R1T2.jpg"
    result = validator.validate_file(invalid_file)
    print(f"File: {invalid_file.name}")
    print(f"Valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_message}")
    
    # Validate the entire directory
    print("\nValidating all files in the directory:")
    results = validator.validate_directory(test_dir)
    
    print(f"Total files: {results['total_files']}")
    print(f"Valid files: {results['valid_count']}")
    print(f"Invalid files: {results['invalid_count']}")
    
    # Print details of invalid files
    if results['invalid_count'] > 0:
        print("\nInvalid files details:")
        for file_info in results['invalid_files']:
            print(f"- {file_info['name']}: {file_info['result'].error_message}")
    
    # Validate recursively
    print("\nValidating all files recursively:")
    results = validator.validate_directory(test_dir, recursive=True)
    
    print(f"Total files: {results['total_files']}")
    print(f"Valid files: {results['valid_count']}")
    print(f"Invalid files: {results['invalid_count']}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
import shutil
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("file_validation_test")

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.components.raw_file_validation_component import RawFileValidationComponent
from utils.site_data.dm_leaf_site_data import DMLeafSiteData


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
        "DML_20242408_R1T2_extra_info.jpg"
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
        "DML_20242408_R1T2.txt"    # Unsupported extension
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
    """Run the file validation component test."""
    # Set up test directory
    test_dir = Path("./test_files_validation")
    
    # Clean up any existing directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create test files
    create_test_files(test_dir)
    
    # Create site data
    site_data = DMLeafSiteData(collection_date="20242408")
    
    # Create validation component
    validation_component = RawFileValidationComponent(
        verbose=True,
        enable_logging=True,
        enable_console=True,
        recursive=True
    )
    
    # Run the component
    try:
        results = validation_component.run({
            'input_dir': test_dir,
            'site_data': site_data
        })
        
        # Print summary
        print("\nFile Validation Results:")
        print(f"Total files: {results['valid_file_count'] + results['invalid_file_count']}")
        print(f"Valid files: {results['valid_file_count']}")
        print(f"Invalid files: {results['invalid_file_count']}")
        
        # Print valid files
        print("\nValid files:")
        for file_path in results['valid_files']:
            print(f"  - {file_path.name}")
        
        # Print invalid files with reasons
        print("\nInvalid files:")
        for file_info in results['file_validation']['invalid_files']:
            print(f"  - {file_info['name']}: {file_info['result'].error_message}")
            
        # Create report
        report_path = Path("validation_report.json")
        
        # Convert validation results to JSON-serializable format
        report_data = {
            'summary': {
                'total_files': results['valid_file_count'] + results['invalid_file_count'],
                'valid_files': results['valid_file_count'],
                'invalid_files': results['invalid_file_count']
            },
            'valid_files': [str(p) for p in results['valid_files']],
            'invalid_files': [
                {
                    'name': info['name'],
                    'path': info['path'],
                    'error': info['result'].error_message
                }
                for info in results['file_validation']['invalid_files']
            ]
        }
        
        # Write report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\nValidation report written to: {report_path}")
        
    except Exception as e:
        print(f"Error running validation component: {str(e)}")
    finally:
        # Clean up test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    main() 
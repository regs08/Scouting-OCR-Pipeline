#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.managers.setup_manager import SetupManager
from utils.components.directory_creator import DirectoryCreator
from utils.runnable_component_config import RunnableComponentConfig
from utils.path_manager import PathManager
from utils.site_data.dm_leaf_site_data import DMLeafSiteData


def create_test_files(input_dir: Path, site_code: str, date: str, count: int = 3) -> List[str]:
    """
    Create test files with proper naming convention for date extraction.
    
    Args:
        input_dir: Directory to create files in
        site_code: Site code to use in filenames
        date: Date string (YYYYMMDD) to use in filenames
        count: Number of files to create
        
    Returns:
        List of created filenames
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    
    filenames = []
    for i in range(1, count + 1):
        # Create filename with pattern: {site_code}_{date}_R{i}_P{i}.jpg
        filename = f"{site_code}_{date}_R{i}P{i}.jpg"
        file_path = input_dir / filename
        
        # Create an empty file
        with open(file_path, 'w') as f:
            f.write(f"Test file {i}")
            
        filenames.append(filename)
        
    return filenames


def main():
    """
    Example showing date extraction from files during setup.
    
    This demonstrates how the SetupManager:
    1. Extracts dates from filenames in the input directory
    2. Updates the PathManager with the extracted date
    3. Creates directories using the extracted date
    """
    # Set up test input directory
    input_dir = Path("./test_input")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    
    # Create site data
    site_data = DMLeafSiteData()
    
    # Create test files with the specific date format
    test_date = "20242408"  # Special date format YYYYMMDD
    filenames = create_test_files(input_dir, site_data.site_code, test_date)
    
    print(f"\nCreated test files in {input_dir}:")
    for filename in filenames:
        print(f"  - {filename}")
    
    # Create component configuration for the directory creator
    directory_creator_config = RunnableComponentConfig(
        component_class=DirectoryCreator,
        checkpoint_name="create_directories",
        checkpoint_number=1,
        description="Creates the required directory structure",
        metadata={}
    )
    
    # Create setup manager with the directory creator component
    # Note: We don't provide a PathManager - it will be created with the extracted date
    setup_manager = SetupManager(
        component_configs=[directory_creator_config],
        verbose=True,
        enable_logging=True,
        enable_console=True
    )
    
    # Run the setup manager, which will extract the date and create the PathManager
    try:
        result = setup_manager.process({
            'input_dir': input_dir,
            'site_data': site_data
        })
        
        # Get the PathManager from the result
        path_manager = result.get('path_manager')
        
        print(f"\nSetup completed successfully!")
        print(f"Extracted date (batch): {path_manager.batch}")
        print(f"Base directory: {path_manager.base_dir}")
        
        print("\nCreated directories:")
        for name, path in result.get('created_directories', {}).items():
            print(f"  - {name}: {path}")
            
    except Exception as e:
        print(f"Error setting up directories: {str(e)}")
    
    # Clean up test directory
    if input_dir.exists():
        print("\nCleaning up test files...")
        shutil.rmtree(input_dir)


if __name__ == "__main__":
    main() 
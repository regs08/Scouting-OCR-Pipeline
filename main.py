#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.managers.application_manager import ApplicationManager
from utils.runnable_component_config import RunnableComponentConfig
from utils.managers.setup_manager import SetupManager
from utils.components.directory_creator import DirectoryCreator
from utils.site_data.dm_leaf_site_data import DMLeafSiteData
from utils.components.raw_file_validation_component import RawFileValidationComponent
from utils.components.copy_files_to_session import CopyRawFilesToSessionComponent
from utils.components.gt_file_validation_component import GTFileValidationComponent
from utils.components.match_gt_and_raw_component import MatchGTAndRawComponent

# Define directory creator component
directory_setup_config = RunnableComponentConfig(
    component_class=DirectoryCreator,
    checkpoint_name="directory_setup",
    checkpoint_number=1,
    description="Set up directory structure"
)

file_validator_config = RunnableComponentConfig(
    component_class=RawFileValidationComponent,
    checkpoint_name="file_validation",
    checkpoint_number=2,
    description="validate raw files"
)

copy_raw_files_config = RunnableComponentConfig(
    component_class=CopyRawFilesToSessionComponent,
    checkpoint_name="copy_files_to_session",
    checkpoint_number=3,
    description="Copy validated files to session"
)
copy_gt_files_config = RunnableComponentConfig(
    component_class=GTFileValidationComponent,
    checkpoint_name="file_validation",
    checkpoint_number=4,
    description="Copy validate_gt_file to session"
)

match_gt_and_raw_config = RunnableComponentConfig(
    component_class=MatchGTAndRawComponent,
    checkpoint_name="match_gt_and_raw",
    checkpoint_number=5,
    description="Match GT and raw data folders and flag unmatched"
)

# Define setup manager with its components
setup_config = RunnableComponentConfig(
    component_class=SetupManager,
    checkpoint_name="ckpt1_setup",
    checkpoint_number=1,
    description="Initial setup and data validation",
    metadata={
        "setup_type": "initial",
        "requires_validation": True
    },
    component_configs=[
        directory_setup_config,
        file_validator_config,
        copy_raw_files_config,
        copy_gt_files_config,
        match_gt_and_raw_config
    ]
)

# Application manager components
app_manager_components = [setup_config]

def main():
    """
    Main entry point for the OCR pipeline application.
    
    This sets up and runs the application pipeline:
    1. The ApplicationManager starts the pipeline
    2. The SetupManager analyzes files and extracts dates
    3. The DirectoryCreator creates the directory structure using the extracted date
    """
    # Configuration
    input_dir = str(Path.cwd() / "input/data")
    site_data = DMLeafSiteData(collection_date='20241408')  # Use DMLeafSiteData for this example
    
    # Create application manager
    app_manager = ApplicationManager(
        input_dir=input_dir,
        site_data=site_data,
        component_configs=app_manager_components,
        verbose=True,
        enable_logging=True,
        enable_console=True
    )
    
    # Run the application pipeline
    print(f"Starting application pipeline for session {app_manager.session_id}...")
    try:
        result = app_manager.run()  # This executes the complete pipeline
        
        # Print success information
        print("\nApplication pipeline completed successfully!")
        print(f"Session ID: {result.get('session_id', 'Unknown')}")
        
        # Print date extraction and directory information
        if 'path_manager' in result:
            path_manager = result['path_manager']
            print(f"\nExtracted date (batch): {path_manager.batch}")
            print(f"Base directory: {path_manager.base_dir}")
        
        # Print created directories
        if 'created_directories' in result:
            print("\nCreated directories:")
            for name, path in result.get('created_directories', {}).items():
                print(f"  - {name}: {path}")
                
        print("\nProcessing complete.")
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        raise
    
    return result

if __name__ == "__main__":
    main() 


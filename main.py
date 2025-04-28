#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.managers.application_manager import ApplicationManager
from utils.site_data.dm_leaf_site_data import DMLeafSiteData
from utils.site_data.arget_singer_24 import ArgetSinger24SiteData
from pipelines.setup_manager_pipeline import setup_config
from pipelines.model_manager_pipeline import model_config
from pipelines.validate_processing_pipeline import validate_config

# Create OCR processor component config



# Example pipeline configuration for ValidateProcessingManager

    # Add more components as needed

# Application manager components
app_manager_components = [setup_config, model_config, validate_config]

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
    site_data = ArgetSinger24SiteData(collection_date='20241408')  # Using ArgetSinger24 format
    #site_data = DMLeafSiteData(collection_date='20241408')  # Using DMLeaf format
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
        


        # Run model processing
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        raise
    
    return result

if __name__ == "__main__":
    main() 


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
from utils.site_data.test_site_data import TestSiteData
from pipelines.setup_manager_pipeline import setup_config
from pipelines.model_manager_pipeline import model_config
from pipelines.validate_processing_pipeline import validate_config
from pipelines.confusion_matrix_config import confusion_matrix_config



# Application manager components
app_manager_components = [setup_config, model_config, validate_config, confusion_matrix_config] 

def main():
    """
    Main entry point for the OCR pipeline application.
    
    This sets up and runs the application pipeline:
    1. The ApplicationManager starts the pipeline
    2. The SetupManager analyzes files and extracts dates
    3. The DirectoryCreator creates the directory structure using the extracted date
    """
    # Configuration
    working_folder = 'test_data_twos'
    base_dir = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline")
    input_dir = str(base_dir / "input" / working_folder / "data")
    gt_dir = str(base_dir / "input" / working_folder / "ground_truth")
    #site_data = ArgetSinger24SiteData(collection_date='20241408')  # Using ArgetSinger24 format
    #site_data = DMLeafSiteData(collection_date='20241408')  # Using DMLeaf format
    site_data = TestSiteData(collection_date='20250505')  # Using Test format
    # Create application manager
    app_manager = ApplicationManager(
        input_dir=input_dir,
        site_data=site_data,
        component_configs=app_manager_components,
        verbose=True,
        enable_logging=True,
        enable_console=True,
        gt_dir=gt_dir  # Pass the ground truth directory explicitly
    )
    
    # Run the application pipeline
    print(f"Starting application pipeline for session {app_manager.session_id}...")
    try:
        result = app_manager.run()  # This executes the complete pipeline
        
        # Print success information
        print("\nApplication pipeline completed successfully!")
        print(f"Session ID: {result.get('session_id', 'Unknown')}")
        

        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        raise
    
    return result

if __name__ == "__main__":
    main() 


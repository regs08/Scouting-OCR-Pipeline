#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.managers.application_manager import ApplicationManager
from utils.managers.session.leaf_cluster_session_manager import LeafClusterSessionManager

from utils.component_config import RunnableComponentConfig
from utils.managers.setup_manager import SetupManager
from utils.setup_processors.directory_setup_processor import DirectorySetupProcessor

from utils.site_data.dm_leaf_site_data import DMLeafSiteData

# Define setup processor components
directory_setup_config = RunnableComponentConfig(
    component_class=DirectorySetupProcessor,
    checkpoint_name="directory_setup",
    checkpoint_number=1,
    description="Set up directory structure"
)

# Define setup manager with its components
setup_config = RunnableComponentConfig(
    component_class=SetupManager,
    checkpoint_name="ckpt1_setup",
    checkpoint_number=1,
    description="Initial setup and data validation",
    component_configs=[directory_setup_config]  # Pass setup processors as nested components
)

# Application manager components
app_manager_components = [setup_config]

def main():
    # Configuration
    input_dir = str(Path.cwd() / "input")
    site_data = DMLeafSiteData()
    
    # Create application manager which internally sets up the setup and session managers
    app_manager = ApplicationManager(
        input_dir=input_dir,
        verbose=True,
        component_configs=app_manager_components,
        site_data=site_data
    )
    
    # Run the complete pipeline - no input_data needed as it uses manager's state
    print(f"Starting application for session {app_manager.session_id}...")
    result = app_manager.run()  # Uses BaseManager's run method
    
    # result now contains both checkpoint_status and output_data
    print(f"Processing complete. Checkpoint status: {result['checkpoint_status']}")
    if 'error' in result:
        print(f"Error occurred: {result['error']}")
    
    return result

if __name__ == "__main__":
    main() 


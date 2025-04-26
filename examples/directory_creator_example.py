import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.managers.setup_manager import SetupManager
from utils.components.directory_creator import DirectoryCreator
from utils.runnable_component_config import RunnableComponentConfig
from utils.path_manager import PathManager
from utils.site_data.site_data_base import SiteDataBase


class ExampleSiteData(SiteDataBase):
    """Example site data for demonstration"""
    def __init__(self):
        data_cols = ['Col1', 'Col2', 'Col3']
        index_cols = ['Row', 'Panel']
        
        super().__init__(
            data_cols=data_cols,
            index_cols=index_cols,
            location_pattern=r'R(\d+)P(\d+)',
            site_name="example_site",
            site_code="EX01"
        )
        self.date = datetime.now().strftime("%Y%m%d")
        
    def get_data_column_indices(self) -> List[int]:
        """Get indices of data columns"""
        return list(range(len(self.data_cols)))
    
    def get_index_column_indices(self) -> List[int]:
        """Get indices of index columns"""
        return list(range(len(self.index_cols)))


def main():
    """Example of using the DirectoryCreator with SetupManager"""
    # Define input directory
    input_dir = Path("./input_data")
    input_dir.mkdir(exist_ok=True)
    
    # Create site data
    site_data = ExampleSiteData()
    
    # Create path manager
    path_manager = PathManager(
        expected_site_code=site_data.site_code,
        batch=datetime.now().strftime("%Y%m%d")
    )
    
    # Create component configuration for the directory creator
    directory_creator_config = RunnableComponentConfig(
        component_class=DirectoryCreator,
        checkpoint_name="create_directories",
        checkpoint_number=1,
        description="Creates the required directory structure",
        metadata={}
    )
    
    # Create setup manager with the directory creator component
    setup_manager = SetupManager(
        component_configs=[directory_creator_config],
        path_manager=path_manager,  # Pass the path manager to setup manager
        verbose=True,
        enable_logging=True,
        enable_console=True
    )
    
    # Run the setup manager
    try:
        result = setup_manager.process({
            'input_dir': input_dir,
            'site_data': site_data,
            'path_manager': path_manager  # Also include in the process input
        })
        
        print("\nSetup completed successfully!")
        print(f"Session ID: {result.get('session_id', 'Unknown')}")
        print("\nCreated directories:")
        for name, path in result.get('created_directories', {}).items():
            print(f"  - {name}: {path}")
            
    except Exception as e:
        print(f"Error setting up directories: {str(e)}")


if __name__ == "__main__":
    main() 
from .site_data_base import SiteDataBase
from typing import List

class ArgetSinger24SiteData(SiteDataBase):
    """Site data configuration for Arget Singer 2024 format."""
    
    def __init__(self):
        """Initialize with fixed column definitions for Arget Singer 2024 format."""
        data_cols = [
            'L1', 'L2', 'L3', 'L4', 'L5', 
            'L6', 'L7', 'L8', 'L9', 'L10', 
            'L11', 'L12', 'L13', 'L14', 'L15', 
            'L16', 'L17', 'L18', 'L19', 'L20'
        ]
        
        index_cols = ['date', 'row', 'panel']
        
        super().__init__(
            data_cols=data_cols,
            index_cols=index_cols,
            site_name="arget_singer_2024",
            site_code="AS24",
            location_pattern=r'R(\d+)P(\d+)',  # Standard R#P# pattern
            supported_extensions=['.png', '.jpg', '.jpeg', '.pdf']
        )
    
    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        
        Returns:
            List of column indices containing data
        """
        return [i for i, col in enumerate(self.data_cols) if col.startswith('L')]
    
    def get_index_column_indices(self) -> List[int]:
        """
        Get the indices of columns used as indices.
        
        Returns:
            List of column indices for date, row, and panel
        """
        return [i for i, col in enumerate(self.index_cols)]
    

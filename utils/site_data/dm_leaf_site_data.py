from .site_data_base import SiteDataBase
from typing import List

class DMLeafSiteData(SiteDataBase):
    """Site data configuration for DM Leaf format."""
    
    def __init__(self):
        """Initialize with fixed column definitions for DM Leaf format."""
        data_cols = [
            'L1', 'L2', 'L3', 'L4', 'L5', 
            'L6', 'L7', 'L8', 'L9', 'L10', 
            'L11', 'L12', 'L13', 'L14', 'L15', 
            'L16', 'L17', 'L18', 'L19', 'L20'
        ]
        
        index_cols = ['Treatment', 'Rep']
        
        super().__init__(
            data_cols=data_cols,
            index_cols=index_cols,
            site_name="dm_leaf",
            site_code="DML",
            location_pattern=r'R(\d+)T(\d+)',  # Standard R#T# pattern for replicate and treatment
            supported_extensions=['.png', '.jpg', '.jpeg', '.pdf']
        )
    
    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        
        Returns:
            List of column indices containing leaf data
        """
        return [i for i, col in enumerate(self.data_cols) if col.startswith('L')]
    
    def get_index_column_indices(self) -> List[int]:
        """
        Get the indices of columns used as indices.
        
        Returns:
            List of column indices for Treatment and Rep
        """
        return [i for i, col in enumerate(self.index_cols)]

from .site_data_base import SiteDataBase
from typing import List, Optional
from datetime import datetime

class DMClusterSiteData(SiteDataBase):
    """Site data configuration for DM Cluster format."""
    
    def __init__(self, collection_date: Optional[str] = None):
        """
        Initialize with fixed column definitions for DM Cluster format.
        
        Args:
            collection_date: Collection date in YYYYMMDD format. If None, current date is used.
        """
        data_cols = [
            'C1', 'C2', 'C3', 'C4', 'C5', 
            'C6', 'C7', 'C8', 'C9', 'C10', 
            'C11', 'C12', 'C13', 'C14', 'C15', 
            'C16', 'C17', 'C18', 'C19', 'C20'
        ]
        
        index_cols = ['Treatment', 'Rep']
        
        super().__init__(
            data_cols=data_cols,
            index_cols=index_cols,
            site_name="dm_cluster",
            site_code="DMC",
            location_pattern=r'R(\d+)T(\d+)',  # Standard R#T# pattern for replicate and treatment
            supported_extensions=['.png', '.jpg', '.jpeg', '.pdf'],
            collection_date=collection_date
        )
    
    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        
        Returns:
            List of column indices containing cluster data
        """
        return [i for i, col in enumerate(self.data_cols) if col.startswith('C')]
    
    def get_index_column_indices(self) -> List[int]:
        """
        Get the indices of columns used as indices.
        
        Returns:
            List of column indices for Treatment and Rep
        """
        return [i for i, col in enumerate(self.index_cols)]
    

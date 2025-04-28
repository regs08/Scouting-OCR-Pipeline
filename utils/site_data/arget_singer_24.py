from .site_data_base import SiteDataBase
from typing import List, Optional
from datetime import datetime
import re

class ArgetSinger24SiteData(SiteDataBase):
    """Site data configuration for Arget Singer 2024 format."""
    
    def __init__(self, collection_date: Optional[str] = None):
        """
        Initialize with fixed column definitions for Arget Singer 2024 format.
        
        Args:
            collection_date: Collection date in YYYYMMDD format. If None, current date is used.
        """
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
            site_name="arget_singer",
            site_code="AS",
            location_pattern=r'R\d+P\d+_R\d+P\d+',  # Pattern for full location string
            supported_extensions=['.png', '.jpg', '.jpeg', '.pdf', '.csv'],
            collection_date=collection_date,
            file_pattern=r'^(AS)_(\d{8})_(.+?)$'  # Pattern for entire filename
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
    
    def extract_location_info(self, location_str: str) -> tuple:
        """
        Extract row and panel from the first location in R#P#_R#P# format
        """
        if not location_str:
            return None, None, None
            
        # Extract first R#P# from R#P#_R#P#
        match = re.match(r'R(\d+)P(\d+)', location_str)
        if match:
            row = int(match.group(1))
            panel = int(match.group(2))
            return row, panel, location_str
        return None, None, None
    

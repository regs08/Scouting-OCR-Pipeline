from .base_col_values import ColIdxBaseProcessor
from typing import List

class ArgetSinger24Values(ColIdxBaseProcessor):
    """Static column definitions for Arget Singer 2024 format."""
    
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
            index_cols=index_cols
        )
    
    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        
        Returns:
            List of column indices containing data
        """
        # Example: columns 2-8 contain actual data values
        return list(range(2, 9))
    

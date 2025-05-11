from typing import List
from utils.site_data.site_data_base import SiteDataBase

class TestSiteData(SiteDataBase):
    """
    Site data configuration for test data.
    Uses L1-L20 as data columns and R#P#_R#P# as location pattern.
    """
    
    def __init__(self, collection_date: str = None):
        """
        Initialize test site data configuration.
        
        Args:
            collection_date: Collection date in YYYYMMDD format. If None, uses current date.
        """
        # Define data columns L1 through L20
        data_cols = [f'L{i}' for i in range(1, 21)]
        
        # Define index columns (empty for test data)
        index_cols = ['R.P', 'Date', 'Rep', 'TRT', 'Path']
        
        # Location pattern: R#P#_R#P# format
        location_pattern = r'R(\d+)P(\d+)_R(\d+)P(\d+)'
        
        # Custom file pattern for test data: TEST_YYYYMMDD_R#P#_R#P#.ext
        file_pattern = r'^(TEST)_(\d{8})_(R\d+P\d+_R\d+P\d+).*$'
        
        super().__init__(
            data_cols=data_cols,
            index_cols=index_cols,
            location_pattern=location_pattern,
            site_name="Test",
            collection_date=collection_date,
            site_code="TEST",
            file_pattern=file_pattern
        )
    
    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        
        Returns:
            List of column indices (0-19 for L1-L20)
        """
        return list(range(20))  # 0-19 for L1-L20
    
    def get_index_column_indices(self) -> List[int]:
        """
        Get the indices of columns used as indices.
        
        Returns:
            Empty list as test data has no index columns
        """
        return [] 
from typing import List

class ColIdxBaseProcessor:
    """Base class for column index value processing."""
    
    def __init__(self, data_cols: List[str], index_cols: List[str]):
        self.data_cols = data_cols
        self.index_cols = index_cols

    def get_data_column_indices(self) -> List[int]:
        """
        Get the indices of columns that contain actual data.
        Must be implemented by child classes.
        """
        raise NotImplementedError

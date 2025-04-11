from .static_cols_base import ColIdxBaseProcessor

class ArgetSinger24ColIdxProcessor(ColIdxBaseProcessor):
    """Processor for Arget Singer 24 data format."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the processor with specific column definitions.
        
        Args:
            verbose: Whether to display detailed information
        """
        super().__init__(verbose)
        
        # Define column types for Arget Singer 24 format
        self.data_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 
                         'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20']
        self.index_cols = ['date', 'row', 'panel']

import pandas as pd
from typing import Optional

class BaseProcessor:
    """Base class for all data processors."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the base processor.
        
        Args:
            verbose: Whether to display detailed information
        """
        self.verbose = verbose

    def display_info(self, df: pd.DataFrame, title: str = "DataFrame Info") -> None:
        """Display information about a DataFrame.
        
        Args:
            df: DataFrame to display information about
            title: Title for the information section
        """
        if self.verbose:
            print(f"\n=== {title} ===")
            print(f"Shape: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())

import pandas as pd
from pathlib import Path
from typing import Union
from .base import BaseProcessor

class DataLoader(BaseProcessor):
    """Processor for loading and basic DataFrame operations."""
    
    def load_df(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a DataFrame from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(file_path)
        self.display_info(df, f"Loaded DataFrame from {file_path}")
        return df

    def process_table(self, table_data: list) -> pd.DataFrame:
        """Process raw table data into a DataFrame.
        
        Args:
            table_data: Raw table data as 2D list
            
        Returns:
            Processed DataFrame
        """
        df = pd.DataFrame(table_data)
        self.display_info(df, "Raw Table Data")
        return df

    def reset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reset the index of a DataFrame.
        
        Args:
            df: DataFrame to reset index for
            
        Returns:
            DataFrame with reset index
        """
        df = df.copy()
        df.columns = df.iloc[0]  # Set first row as column headers
        df = df[1:]             # Drop the now-header row from data
        df.reset_index(drop=True, inplace=True)  # Reset row index
        self.display_info(df, "After Index Reset")
        return df 
import pandas as pd
from pathlib import Path
from typing import Union
from utils.base_processor import BaseProcessor

class DataLoader(BaseProcessor):
    """Processor for loading and basic DataFrame operations."""
    
    def load_df(self, file_path: Union[str, Path], reset_index: bool = False) -> pd.DataFrame:
        """Load a DataFrame from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(file_path)
        if reset_index:
            df = self.reset_index(df)
        
        # Use display_and_log to log the loading of the DataFrame
        if self.verbose:
            self.display_and_log(f"Loaded DataFrame from {file_path}", {"Shape": df.shape})
        return df

    def process_table(self, table_data: list) -> pd.DataFrame:
        """Process raw table data into a DataFrame.
        
        Args:
            table_data: Raw table data as 2D list
            
        Returns:
            Processed DataFrame
        """
        df = pd.DataFrame(table_data)
        
        # Use display_and_log to log the raw table data
        self.display_and_log("Raw Table Data", {"Data": df.head().to_dict()})
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
        df = df[1:]               # Drop the now-header row from data
        df.reset_index(drop=True, inplace=True)  # Reset row index
        
        # Use display_and_log to log the DataFrame after index reset
        if self.verbose:
            self.display_and_log("After Index Reset", {"Shape": df.shape, "Columns": df.columns.tolist()})
        return df 
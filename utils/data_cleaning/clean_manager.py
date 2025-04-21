import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from utils.base_manager import BaseManager
from utils.path_manager import PathManager
from utils.data_cleaning.basic_cleaner import BasicCleaner
from utils.data_cleaning.simple_value_cleaner import SimpleValueCleaner

class CleanManager(BaseManager):
    """Manager for orchestrating data cleaning operations on DataFrames."""
    
    def __init__(self,
                 path_manager: PathManager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the CleanManager.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name="clean_manager"
        )
        
        # Initialize cleaning components
        self.basic_cleaner = BasicCleaner(verbose=verbose, enable_logging=enable_logging)
        self.simple_cleaner = SimpleValueCleaner(verbose=verbose, enable_logging=enable_logging)
        
        # Add components to pipeline
        self.add_component(self.basic_cleaner, "basic_cleaning", 1)
        self.add_component(self.simple_cleaner, "value_cleaning", 2)
        
    def convert_to_integers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert specified columns to integers, handling non-numeric values.
        
        Args:
            df: DataFrame to process
            columns: List of columns to convert. If None, converts all columns.
            
        Returns:
            DataFrame with converted columns
        """
        df = df.copy()
        
        # If no columns specified, use all columns
        if columns is None:
            columns = df.columns.tolist()
        else:
            # Filter to only include columns that actually exist in the DataFrame
            columns = [col for col in columns if col in df.columns]
            
        self.log_info("convert_to_integers", f"Converting {len(columns)} columns to integers")
        
        # Track conversion statistics
        conversion_stats = {
            "successful": 0,
            "failed": 0,
            "total": 0
        }
        
        for col in columns:
            try:
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # Count successful and failed conversions
                successful = numeric_series.notna().sum()
                failed = numeric_series.isna().sum()
                
                conversion_stats["successful"] += successful
                conversion_stats["failed"] += failed
                conversion_stats["total"] += len(df[col])
                
                # Convert to integers, replacing NaN with 0
                df[col] = numeric_series.fillna(0).astype(int)
                
                self.log_info("convert_to_integers", 
                            f"Column '{col}': {successful} successful, {failed} failed conversions")
                
            except Exception as e:
                self.log_error("convert_to_integers", 
                             f"Error converting column '{col}': {str(e)}")
                continue
                
        # Log overall statistics
        self.log_info("convert_to_integers", "Conversion statistics", conversion_stats)
        
        return df
        
    def clean_dataframe(self, 
                       df: pd.DataFrame,
                       numeric_cols: Optional[List[str]] = None,
                       replace_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Clean a DataFrame using all available cleaning components.
        
        Args:
            df: DataFrame to clean
            numeric_cols: List of columns to process for numeric cleaning
            replace_dict: Dictionary mapping values to replace with their replacements
            
        Returns:
            Cleaned DataFrame
        """
        # Start with basic cleaning
        cleaned_df = self.basic_cleaner.clean_zeros(df, numeric_cols)
        
        # Apply value replacements if specified
        if replace_dict:
            cleaned_df = self.simple_cleaner.replace_values(cleaned_df, replace_dict)
            
        # Convert to integers
        cleaned_df = self.convert_to_integers(cleaned_df, numeric_cols)
            
        return cleaned_df
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through the cleaning pipeline.
        
        Args:
            input_data: Dictionary containing:
                - 'dataframes': Dictionary mapping filenames to DataFrames
                - 'numeric_cols': Optional list of columns to process
                - 'replace_dict': Optional dictionary of value replacements
                
        Returns:
            Dictionary containing:
                - 'cleaned_dataframes': Dictionary mapping filenames to cleaned DataFrames
                - 'cleaning_stats': Dictionary of cleaning statistics
        """
        try:
            dataframes = input_data.get('dataframes', {})
            numeric_cols = input_data.get('numeric_cols')
            replace_dict = input_data.get('replace_dict')
            
            if not dataframes:
                self.log_error("process", "No dataframes provided for cleaning")
                return {"error": "No dataframes provided"}
                
            cleaned_dataframes = {}
            cleaning_stats = {}
            
            for filename, df in dataframes.items():
                try:
                    # Clean the DataFrame
                    cleaned_df = self.clean_dataframe(df, numeric_cols, replace_dict)
                    cleaned_dataframes[filename] = cleaned_df
                    
                    # Generate cleaning statistics
                    stats = {
                        "original_shape": df.shape,
                        "cleaned_shape": cleaned_df.shape,
                        "changes_made": not df.equals(cleaned_df)
                    }
                    cleaning_stats[filename] = stats
                    
                    # Save cleaned DataFrame
                    output_dir = self.path_manager.get_checkpoint_path(
                        session_id=self.session_id,
                        checkpoint="ckpt5_cleaned"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f"{filename}.csv"
                    cleaned_df.to_csv(output_path, index=False)
                    self.log_info("process", f"Saved cleaned data to: {output_path}")
                    
                except Exception as e:
                    self.log_error("process", f"Error cleaning {filename}: {str(e)}")
                    continue
                    
            return {
                "cleaned_dataframes": cleaned_dataframes,
                "cleaning_stats": cleaning_stats
            }
            
        except Exception as e:
            self.log_error("process", f"Error in cleaning process: {str(e)}")
            return {"error": str(e)} 
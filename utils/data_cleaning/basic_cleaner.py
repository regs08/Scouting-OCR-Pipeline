import pandas as pd
from typing import List, Optional
from utils.base_processor import BaseProcessor

class BasicCleaner(BaseProcessor):
    """
    Simple cleaner for basic data cleaning operations.
    Focuses on cleaning "0" values that are commonly misrecognized by OCR.
    """
    
    def __init__(
        self, 
        verbose: bool = False,
        enable_logging: bool = False
    ):
        """
        Initialize the BasicCleaner.
        
        Args:
            verbose: Whether to display detailed output
            enable_logging: Whether to enable logging
        """
        super().__init__(verbose=verbose, enable_logging=enable_logging)
    
    def clean_zeros(
        self, 
        df: pd.DataFrame, 
        numeric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Simple cleaning function that replaces empty values and :selected: with 0.
        
        Args:
            df: DataFrame to clean
            numeric_cols: List of columns to process. If None, processes all columns.
            
        Returns:
            Cleaned DataFrame with standardized zeros
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Define default numeric columns if none provided
        if numeric_cols is None:
            numeric_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 
                          'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20']
        
        # Filter to only include columns that actually exist in the DataFrame
        numeric_cols = [col for col in numeric_cols if col in cleaned_df.columns]
        
        self.display_and_log(f"Cleaning zeros in {len(numeric_cols)} columns")
        
        # Track replacements for reporting
        replacements = {
            "empty_to_zero": 0,
            "selected_to_zero": 0
        }
        
        # Process each column
        for col in numeric_cols:
            # Replace empty values with "0"
            empty_mask = cleaned_df[col].isna() | (cleaned_df[col].astype(str) == "")
            replacements["empty_to_zero"] += empty_mask.sum()
            cleaned_df.loc[empty_mask, col] = "0"
            
            # Replace ":selected:" with "0"
            selected_mask = cleaned_df[col].astype(str) == ":selected:"
            replacements["selected_to_zero"] += selected_mask.sum()
            cleaned_df.loc[selected_mask, col] = "0"
        
        # Log summary of replacements
        self.display_and_log("Zero cleaning completed", {
            "Empty values replaced": replacements["empty_to_zero"],
            "':selected:' values replaced": replacements["selected_to_zero"],
            "Total replacements": sum(replacements.values())
        })
        
        return cleaned_df 
        
    def replace_values(
        self,
        df: pd.DataFrame,
        replace_dict: dict,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Replace specific values in the DataFrame with provided replacements.
        
        Args:
            df: DataFrame to clean
            replace_dict: Dictionary mapping values to replace with their replacements
            columns: List of columns to process. If None, processes all columns.
            
        Returns:
            Cleaned DataFrame with replaced values
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # If no columns specified, use all columns
        if columns is None:
            columns = cleaned_df.columns.tolist()
        else:
            # Filter to only include columns that actually exist in the DataFrame
            columns = [col for col in columns if col in cleaned_df.columns]
        
        self.display_and_log(f"Replacing values in {len(columns)} columns")
        
        # Track replacements for reporting
        replacements = {str(old_val): 0 for old_val in replace_dict.keys()}
        
        # Process each column
        for col in columns:
            for old_val, new_val in replace_dict.items():
                # Create mask for values to replace
                replace_mask = cleaned_df[col].astype(str) == str(old_val)
                replacements[str(old_val)] += replace_mask.sum()
                
                # Replace the values
                cleaned_df.loc[replace_mask, col] = new_val
        
        # Log summary of replacements
        replacement_details = {f"'{old}' → '{replace_dict[old]}'": count 
                              for old, count in replacements.items() if count > 0}
        replacement_details["Total replacements"] = sum(replacements.values())
        
        self.display_and_log("Value replacement completed", replacement_details)
        
        return cleaned_df 
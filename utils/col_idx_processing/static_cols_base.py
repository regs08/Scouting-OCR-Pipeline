import pandas as pd
from typing import Optional, Tuple, Dict, List
from utils.base_processor import BaseProcessor 
import os
class ColIdxBaseProcessor(BaseProcessor):
    """Base class for all data processors with CSV saving capability."""
        
    def __init__(self, verbose: bool = False, save_path: str = None, enable_logging: bool = False):
        """Initialize the base processor.
        
        Args:
            verbose: Whether to display detailed information
            save_path: Path to save processed data (if applicable)
            enable_logging: Whether to enable logging
        """
        super().__init__(verbose=verbose, enable_logging=enable_logging)  # Call the base class constructor
        self.save_path = save_path
        self.pred_df: Optional[pd.DataFrame] = None
        self.gt_df: Optional[pd.DataFrame] = None
        
        # Column types will be passed in during initialization
        self.data_cols: Optional[list] = None
        self.index_cols: Optional[list] = None

    def set_dataframes(self, gt_df: pd.DataFrame, ocr_df: pd.DataFrame) -> None:
        """Set the ground truth and OCR DataFrames.
        
        Args:
            gt_df: Ground truth DataFrame
            ocr_df: OCR DataFrame
        """
        self.gt_df = gt_df
        self.pred_df = ocr_df
        
        if self.verbose:
            self.display_and_log("\n=== DataFrames Set ===")
            self.display_and_log("Ground Truth DataFrame:", {"Shape": gt_df.shape})
            self.display_and_log("OCR DataFrame:", {"Shape": ocr_df.shape})

    def display_info(self, df: pd.DataFrame, title: str = "DataFrame Info") -> None:
        """Display information about a DataFrame.
        
        Args:
            df: DataFrame to display information about
            title: Title for the information section
        """
        self.display_and_log(f"\n=== {title} ===", {"Shape": df.shape, "Columns": df.columns.tolist()})
        self.display_and_log("First few rows:", {"Data": df.head().to_dict()})
        self.display_and_log("Data Types:", {"Types": df.dtypes.to_dict()})

    def validate_dataframes(self) -> Tuple[bool, str, Dict[str, List[Tuple[str, int]]]]:
        """Validate that both DataFrames are set and have the correct structure.
        
        Returns:
            Tuple of (is_valid, message, column_info)
            - is_valid: True if DataFrames are valid
            - message: Description of any issues found
            - column_info: Dictionary containing extra and missing column information
        """
        if self.gt_df is None or self.pred_df is None:
            return False, "DataFrames not set. Call set_dataframes() first.", {}
            
        # Get all columns from both DataFrames
        gt_cols = set(self.gt_df.columns)
        ocr_cols = set(self.pred_df.columns)
        
        # Find extra and missing columns
        extra_cols = [(col, self.pred_df.columns.get_loc(col)) 
                     for col in ocr_cols - gt_cols]
        missing_cols = [(col, self.gt_df.columns.get_loc(col)) 
                       for col in gt_cols - ocr_cols]
        
        # Build detailed message
        message_parts = []
        
        if extra_cols:
            message_parts.append(f"➕ Extra columns in OCR: {[f'{col} (at index {idx})' for col, idx in extra_cols]}")
            
        if missing_cols:
            message_parts.append(f"➖ Missing columns in OCR: {[f'{col} (at index {idx})' for col, idx in missing_cols]}")
            
        if not extra_cols and not missing_cols and len(gt_cols) == len(ocr_cols):
            message_parts.append("✅ Columns match exactly")
            
        # Define message and is_valid based on validation results    
        message = " | ".join(message_parts) if message_parts else "No column differences found"
        is_valid = not (extra_cols or missing_cols)
        
        # Create column info dictionary
        column_info = {
            'extra_columns': extra_cols,
            'missing_columns': missing_cols
        }
            
        return is_valid, message, column_info

    def rename_columns_by_index(self) -> Tuple[pd.DataFrame, str]:
        """Rename OCR columns to match ground truth columns based on their indices.
        
        Returns:
            Tuple of (processed_df, message)
            - processed_df: DataFrame with renamed columns
            - message: Description of the renaming process
        """
        if self.gt_df is None or self.pred_df is None:
            return self.pred_df, "DataFrames not set. Call set_dataframes() first."
            
        # Get validation results
        _, _, column_info = self.validate_dataframes()
        
        # Create a copy of OCR DataFrame
        processed_df = self.pred_df.copy()
        
        # Create mapping of indices to ground truth column names
        gt_col_mapping = {self.gt_df.columns.get_loc(col): col for col in self.gt_df.columns}
        
        # Create renaming dictionary
        rename_dict = {}
        for ocr_col, idx in column_info['extra_columns']:
            if idx in gt_col_mapping:
                rename_dict[ocr_col] = gt_col_mapping[idx]
        
        # Rename columns
        processed_df = processed_df.rename(columns=rename_dict)
        
        # Build message
        if rename_dict:
            message = f"Renamed columns: {[f'{old} → {new}' for old, new in rename_dict.items()]}"
        else:
            message = "No columns needed renaming"
            
        if self.verbose:
            self.display_and_log("\n=== Column Renaming ===")
            self.display_and_log(message)
            self.display_and_log("Before renaming:", {"Columns": self.pred_df.columns.tolist()})
            self.display_and_log("After renaming:", {"Columns": processed_df.columns.tolist()})
        
        self.pred_df = processed_df
        return processed_df, message

    def save_to_csv(self, df: pd.DataFrame, filepath: str, outdir: str, prefix: str = "col_matched_") -> None:
        """Save DataFrame to CSV file with column matching prefix.
        
        Args:
            df: DataFrame to save
            filepath: Path where CSV should be saved
            prefix: Prefix to add to filename (default: 'col_matched_')
        """
        try:
            # Add prefix to filename
            new_filepath = os.path.join(outdir, f"{prefix}{os.path.basename(filepath)}")
  
            df.to_csv(os.path.join(outdir, new_filepath), index=False)
            if self.verbose:
                self.display_and_log(f"Successfully saved DataFrame to {new_filepath}")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {str(e)}")
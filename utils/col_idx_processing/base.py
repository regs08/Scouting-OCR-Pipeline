import pandas as pd
from typing import Optional, Tuple, Dict, List

class ColIdxBaseProcessor:
    """Base class for all data processors with CSV saving capability."""
        
    def __init__(self, verbose: bool = False, save_path: str = None):
        """Initialize the base processor.
        
        Args:
            verbose: Whether to display detailed information
        """
        self.verbose = verbose
        self.ocr_df: Optional[pd.DataFrame] = None
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
        self.ocr_df = ocr_df
        
        if self.verbose:
            print("\n=== DataFrames Set ===")
            print("Ground Truth DataFrame:")
            print(f"Shape: {gt_df.shape}")
            print("\nOCR DataFrame:")
            print(f"Shape: {ocr_df.shape}")

    def display_info(self, df: pd.DataFrame, title: str = "DataFrame Info") -> None:
        """Display information about a DataFrame.
        
        Args:
            df: DataFrame to display information about
            title: Title for the information section
        """
        if self.verbose:
            print(f"\n=== {title} ===")
            print(f"Shape: {df.shape}")
            print("\nColumns:")
            for idx, col in enumerate(df.columns):
                print(f"{idx}: {col}")
            print("\nFirst few rows:")
            print(df.head())
            print("\nData Types:")
            print(df.dtypes)

    def validate_dataframes(self) -> Tuple[bool, str, Dict[str, List[Tuple[str, int]]]]:
        """Validate that both DataFrames are set and have the correct structure.
        
        Returns:
            Tuple of (is_valid, message, column_info)
            - is_valid: True if DataFrames are valid
            - message: Description of any issues found
            - column_info: Dictionary containing extra and missing column information
        """
        if self.gt_df is None or self.ocr_df is None:
            return False, "DataFrames not set. Call set_dataframes() first.", {}
            
        # Get all columns from both DataFrames
        gt_cols = set(self.gt_df.columns)
        ocr_cols = set(self.ocr_df.columns)
        
        # Find extra and missing columns
        extra_cols = [(col, self.ocr_df.columns.get_loc(col)) 
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
        if self.gt_df is None or self.ocr_df is None:
            return self.ocr_df, "DataFrames not set. Call set_dataframes() first."
            
        # Get validation results
        _, _, column_info = self.validate_dataframes()
        
        # Create a copy of OCR DataFrame
        processed_df = self.ocr_df.copy()
        
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
            print("\n=== Column Renaming ===")
            print(message)
            print("\nBefore renaming:")
            print(self.ocr_df.columns.tolist())
            print("\nAfter renaming:")
            print(processed_df.columns.tolist())
        self.ocr_df = processed_df
        return processed_df, message
    def save_to_csv(self, df: pd.DataFrame, filepath: str, prefix: str = "col_matched_") -> None:
        """Save DataFrame to CSV file with column matching prefix.
        
        Args:
            df: DataFrame to save
            filepath: Path where CSV should be saved
            prefix: Prefix to add to filename (default: 'col_matched_')
        """
        try:
            # Add prefix to filename
            path_parts = filepath.rsplit('/', 1)
            if len(path_parts) > 1:
                new_filepath = f"{path_parts[0]}/{prefix}{path_parts[1]}"
            else:
                new_filepath = f"{prefix}{filepath}"
                
            df.to_csv(new_filepath, index=False)
            if self.verbose:
                print(f"Successfully saved DataFrame to {new_filepath}")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {str(e)}")
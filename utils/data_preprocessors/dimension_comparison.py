import pandas as pd
from typing import Tuple
from utils.base_processor import BaseProcessor

class DimensionComparison(BaseProcessor):
    """Processor for handling static column operations and comparisons."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the static column processor.
        
        Args:
            verbose: Whether to display detailed information
        """
        super().__init__(verbose)

    def compare_dimensions(self, gt_df: pd.DataFrame, ocr_df: pd.DataFrame) -> Tuple[bool, str]:
        """Compare dimensions between ground truth and OCR DataFrames.
        
        Args:
            gt_df: Ground truth DataFrame
            ocr_df: OCR DataFrame
            
        Returns:
            Tuple of (is_valid, message)
            - is_valid: True if dimensions are compatible
            - message: Description of any issues found
        """
        gt_rows, gt_cols = gt_df.shape
        ocr_rows, ocr_cols = ocr_df.shape
        
        if self.verbose:
            print("\n=== Dimension Comparison ===")
            print(f"Ground Truth: {gt_rows} rows × {gt_cols} columns")
            print(f"OCR Data: {ocr_rows} rows × {ocr_cols} columns")
        
        # Check if column counts match
        if gt_cols != ocr_cols:
            return False, f"Column count mismatch: GT={gt_cols}, OCR={ocr_cols}"
        
        # Check if row counts match
        if gt_rows != ocr_rows:
            return False, f"Row count mismatch: GT={gt_rows}, OCR={ocr_rows}"
        
        return True, "Dimensions match"

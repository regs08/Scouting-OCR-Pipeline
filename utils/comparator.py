from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import re
from abc import ABC, abstractmethod
from utils.base_processor import BaseProcessor

class DataColumnComparator(BaseProcessor):
    """Comparator for data columns (e.g., L1-L20)."""
    
    def __init__(self, data_columns: Optional[List[str]] = None, 
                 validation_rules: Optional[Dict[str, Any]] = None,
                 case_sensitive: bool = False,
                 verbose: bool = False,
                 enable_logging: bool = False):
        """Initialize the DataColumnComparator with configuration.
        
        Args:
            data_columns: List of data columns to compare
            validation_rules: Dictionary of validation rules
            case_sensitive: Whether to perform case-sensitive comparisons
            verbose: Whether to display detailed output
            enable_logging: Whether to enable logging
        """
        super().__init__(verbose=verbose, enable_logging=enable_logging)
        self.data_columns = data_columns or []
        self.validation_rules = validation_rules or {}
        self.case_sensitive = case_sensitive

    def create_confusion_matrix(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame, 
                              columns: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
        """Create a confusion matrix for the OCR results.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            columns: List of columns to analyze
            max_rows: Maximum number of rows to analyze
            
        Returns:
            DataFrame containing confusion matrix
        """
        try:
            self.display_and_log("Creating confusion matrix...", {
                "Columns": columns,
                "Max Rows": max_rows or "All"
            })

            # Reset index for clean access
            ocr_df = ocr_df[columns].reset_index(drop=True).fillna("").astype(str)
            gt_df = gt_df[columns].reset_index(drop=True).fillna("").astype(str)

            row_limit = max_rows or min(len(ocr_df), len(gt_df))
            
            # Initialize confusion matrix dictionary
            confusion_dict = defaultdict(lambda: defaultdict(int))
            
            # Count occurrences of each value pair
            for i in range(row_limit):
                for col in columns:
                    val_ocr = ocr_df.at[i, col]
                    val_gt = gt_df.at[i, col]
                    if not self.case_sensitive:
                        val_ocr = val_ocr.lower()
                        val_gt = val_gt.lower()
                    confusion_dict[val_gt][val_ocr] += 1
            
            # Convert to DataFrame
            confusion_df = pd.DataFrame.from_dict(confusion_dict, orient='index')
            confusion_df = confusion_df.fillna(0)
            
            # Sort index and columns
            confusion_df = confusion_df.sort_index()
            confusion_df = confusion_df.reindex(sorted(confusion_df.columns), axis=1)
            
            # Log summary statistics
            total_predictions = confusion_df.sum().sum()
            correct_predictions = sum(confusion_df[col][col] for col in confusion_df.columns if col in confusion_df.index)
            accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            self.display_and_log("Confusion matrix created", {
                "Total Predictions": total_predictions,
                "Correct Predictions": correct_predictions,
                "Accuracy": f"{accuracy:.2f}%"
            })
            
            return confusion_df
        except Exception as e:
            error_msg = f"Error creating confusion matrix: {str(e)}"
            self.display_and_log(error_msg)
            raise Exception(error_msg)

    def compare_values(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame, 
                      columns: List[str], max_rows: Optional[int] = None) -> Dict[str, Any]:
        """Compare values between OCR and ground truth dataframes.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            columns: List of columns to compare
            max_rows: Maximum number of rows to analyze
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            self.display_and_log("Starting value comparison...", {
                "Columns": columns,
                "Max Rows": max_rows or "All"
            })
            
            mismatches = []
            matches = 0

            # Reset index for clean access
            ocr_df = ocr_df[columns].reset_index(drop=True).fillna("").astype(str)
            gt_df = gt_df[columns].reset_index(drop=True).fillna("").astype(str)

            row_limit = max_rows or min(len(ocr_df), len(gt_df))
            total_cells = row_limit * len(columns)

            for i in range(row_limit):
                for col in columns:
                    val_ocr = ocr_df.at[i, col]
                    val_gt = gt_df.at[i, col]
                    if not self.case_sensitive:
                        val_ocr = val_ocr.lower()
                        val_gt = val_gt.lower()
                    if val_ocr != val_gt:
                        mismatches.append({
                            "row": i,
                            "column": col,
                            "ocr_value": ocr_df.at[i, col],  # Original case
                            "truth_value": gt_df.at[i, col],  # Original case
                            "error_type": "value_mismatch"
                        })
                    else:
                        matches += 1

            accuracy = (matches / total_cells * 100) if total_cells > 0 else 0

            results = {
                "accuracy": accuracy,
                "total_cells": total_cells,
                "matches": matches,
                "mismatches": mismatches
            }

            self.display_and_log("Value comparison completed", {
                "Accuracy": f"{accuracy:.2f}%",
                "Matches": f"{matches}/{total_cells}",
                "Mismatches": len(mismatches)
            })

            return results
        except Exception as e:
            error_msg = f"Error comparing values: {str(e)}"
            self.display_and_log(error_msg)
            raise Exception(error_msg)

    def run_comparison(self, ocr_df: pd.DataFrame, gt_path: str, verbose: bool = False) -> Dict[str, Any]:
        """Run a complete comparison between OCR and ground truth data.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_path: Path to ground truth CSV file
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Load ground truth data
            gt_df = self.load_ground_truth(gt_path)
            ocr_df = ocr_df.fillna("").astype(str)

            # Step 1: Compare columns
            column_comparison = self.compare_columns(ocr_df, gt_df, verbose)

            # Step 2: Compare row count
            row_limit = self.compare_row_counts(ocr_df, gt_df, verbose)

            # Step 3: Compare values
            value_comparison = self.compare_values(
                ocr_df,
                gt_df,
                columns=[col[0] for col in column_comparison["common_columns"]],
                max_rows=row_limit
            )

            # Step 4: Create confusion matrix
            confusion_matrix = self.create_confusion_matrix(
                ocr_df,
                gt_df,
                columns=[col[0] for col in column_comparison["common_columns"]],
                max_rows=row_limit
            )

            return {
                "column_comparison": column_comparison,
                "row_count": row_limit,
                "value_comparison": value_comparison,
                "confusion_matrix": confusion_matrix
            }
        except Exception as e:
            raise Exception(f"Error running comparison: {str(e)}")
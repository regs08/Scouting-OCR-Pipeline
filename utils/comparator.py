from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import re
from abc import ABC, abstractmethod

class BaseComparator(ABC):
    """Base class for all comparators."""
    
    def __init__(self, columns: Optional[List[str]] = None, validation_rules: Optional[Dict[str, Any]] = None):
        """Initialize the base comparator.
        
        Args:
            columns: List of columns to compare
            validation_rules: Dictionary of validation rules for specific columns
        """
        self.columns = columns or []
        self.validation_rules = validation_rules or {}
        
    def load_ground_truth(self, gt_path: str) -> pd.DataFrame:
        """Load ground truth data from CSV file.
        
        Args:
            gt_path: Path to ground truth CSV file
            
        Returns:
            DataFrame containing ground truth data
            
        Raises:
            Exception: If file cannot be loaded
        """
        try:
            return pd.read_csv(gt_path, dtype=str, encoding="ISO-8859-1").fillna("")
        except Exception as e:
            raise Exception(f"Error loading ground truth data: {str(e)}")
    
    def compare_columns(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame, verbose: bool = False) -> Dict[str, Any]:
        """Compare column structure between OCR and ground truth dataframes.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing comparison results
        """
        # Filter for relevant columns only
        ocr_cols = [col for col in ocr_df.columns if col in self.columns]
        gt_cols = [col for col in gt_df.columns if col in self.columns]

        cols_ocr = set(ocr_cols)
        cols_gt = set(gt_cols)

        extra_keys = cols_ocr - cols_gt
        missing_keys = cols_gt - cols_ocr
        common_keys = cols_ocr & cols_gt

        if verbose:
            print(f"📊 {self.__class__.__name__} Column Comparison:")
            print(f"  🔢 Columns in OCR: {len(ocr_cols)}")
            print(f"  🔢 Columns in Ground Truth: {len(gt_cols)}")

        extra_columns = [(col, ocr_df.columns.get_loc(col)) for col in extra_keys]
        missing_columns = [(col, gt_df.columns.get_loc(col)) for col in missing_keys]

        if verbose:
            if extra_columns:
                print(f"  ➕ Extra columns in OCR: {extra_columns}")
            if missing_columns:
                print(f"  ➖ Missing columns in OCR: {missing_columns}")
            if not extra_columns and not missing_columns and len(ocr_cols) == len(gt_cols):
                print("  ✅ Columns match exactly.")

        # Common columns with index info
        common_columns_info = []
        for col in sorted(common_keys):
            idx_ocr = ocr_df.columns.get_loc(col)
            idx_gt = gt_df.columns.get_loc(col)
            common_columns_info.append((col, idx_ocr, idx_gt))

        return {
            "ocr_col_count": len(ocr_cols),
            "truth_col_count": len(gt_cols),
            "extra_columns": extra_columns,
            "missing_columns": missing_columns,
            "common_columns": common_columns_info
        }

    def compare_row_counts(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame, verbose: bool = False) -> int:
        """Compare row counts between OCR and ground truth dataframes.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            verbose: Whether to print detailed output
            
        Returns:
            Minimum row count between OCR and ground truth
        """
        len_ocr, len_gt = len(ocr_df), len(gt_df)

        if verbose:
            print("\n📏 Row Count Comparison:")
            print(f"  OCR Rows: {len_ocr}")
            print(f"  Ground Truth Rows: {len_gt}")

            if len_ocr != len_gt:
                print("  ⚠️ Row count mismatch!")
            else:
                print("  ✅ Row counts match.")

        return min(len_ocr, len_gt)  # use the minimum for safe comparison

    def calculate_accuracy(self, correct_entries: int, total_entries: int) -> float:
        """Calculate accuracy percentage.
        
        Args:
            correct_entries: Number of correct entries
            total_entries: Total number of entries
            
        Returns:
            Accuracy percentage
        """
        return (correct_entries / total_entries * 100) if total_entries > 0 else 0

    @abstractmethod
    def validate_column(self, ocr_series: pd.Series, gt_series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Validate a specific column. To be implemented by child classes.
        
        Args:
            ocr_series: Series containing OCR values
            gt_series: Series containing ground truth values
            column_name: Name of the column being validated
            
        Returns:
            Dictionary containing validation results
        """
        pass

    @abstractmethod
    def analyze_columns(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze columns. To be implemented by child classes.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            
        Returns:
            Dictionary containing analysis results
        """
        pass

    def get_validation_rules(self, column_name: str) -> Dict[str, Any]:
        """Get validation rules for a specific column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dictionary containing validation rules for the column
        """
        return self.validation_rules.get(column_name, {})

class DataColumnComparator(BaseComparator):
    """Comparator for data columns (e.g., L1-L20)."""
    
    def __init__(self, data_columns: Optional[List[str]] = None, 
                 validation_rules: Optional[Dict[str, Any]] = None,
                 case_sensitive: bool = False):
        """Initialize the DataColumnComparator with configuration.
        
        Args:
            data_columns: List of data columns to compare
            validation_rules: Dictionary of validation rules
            case_sensitive: Whether to perform case-sensitive comparisons
        """
        super().__init__(columns=data_columns, validation_rules=validation_rules)
        self.columns = data_columns or ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 
                                      'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20']
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
            
            return confusion_df
        except Exception as e:
            raise Exception(f"Error creating confusion matrix: {str(e)}")

    def compare_values(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame, 
                      columns: List[str], max_rows: Optional[int] = None, 
                      verbose: bool = False) -> Dict[str, Any]:
        """Compare values between OCR and ground truth dataframes.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            columns: List of columns to compare
            max_rows: Maximum number of rows to analyze
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            if verbose:
                print("\n🔍 Value Comparison:")
            
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

            accuracy = self.calculate_accuracy(matches, total_cells)

            if verbose:
                print(f"  ✅ Matching Cells: {matches} / {total_cells}")
                print(f"  🎯 Accuracy: {accuracy:.2f}%")

            return {
                "accuracy": accuracy,
                "total_cells": total_cells,
                "matches": matches,
                "mismatches": mismatches
            }
        except Exception as e:
            raise Exception(f"Error comparing values: {str(e)}")

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
                max_rows=row_limit,
                verbose=verbose
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

    def validate_column(self, ocr_series: pd.Series, gt_series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Validate a data column.
        
        Args:
            ocr_series: Series containing OCR values
            gt_series: Series containing ground truth values
            column_name: Name of the column being validated
            
        Returns:
            Dictionary containing validation results
        """
        # Data columns don't need specific validation
        return None

    def analyze_columns(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data columns.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            
        Returns:
            Dictionary containing analysis results
        """
        return self.run_comparison(ocr_df, gt_df)

class IndexColumnComparator(BaseComparator):
    """Comparator for index columns (e.g., date, row, panel, disease)."""
    
    def __init__(self, index_columns: Optional[List[str]] = None, 
                 validation_rules: Optional[Dict[str, Any]] = None):
        """Initialize the IndexColumnComparator with configuration.
        
        Args:
            index_columns: List of index columns to validate
            validation_rules: Dictionary of validation rules for specific columns
        """
        super().__init__(columns=index_columns, validation_rules=validation_rules)
        self.columns = index_columns or ['date', 'row', 'panel', 'disease']
        
        # Define column indices
        self.column_indices = {
            'date': 0,
            'row': 1,
            'panel': 2,
            'disease': 3
        }
        
        # Default validation rules
        self.validation_rules = validation_rules or {
            'date': {
                'format': '%Y-%m-%d',
                'min_date': '2000-01-01',
                'max_date': '2100-12-31'
            },
            'disease': {
                'allowed_categories': ['DM', 'GLRV', 'Other'],
                'case_sensitive': False
            },
            'panel': {
                'min_value': 1,
                'max_value': 30,
                'sequence_check': True
            }
        }

    def validate_date_column(self, ocr_series: pd.Series, gt_series: pd.Series) -> Dict[str, Any]:
        """Validate date format and accuracy."""
        try:
            results = {
                'format_errors': [],
                'range_errors': [],
                'total_entries': len(ocr_series),
                'correct_entries': 0,
                'accuracy': 0.0,
                'ocr_values': ocr_series.tolist(),
                'gt_values': gt_series.tolist()
            }
            
            rules = self.get_validation_rules('date')
            min_date = datetime.strptime(rules['min_date'], rules['format'])
            max_date = datetime.strptime(rules['max_date'], rules['format'])
            
            for idx, (ocr_date, gt_date) in enumerate(zip(ocr_series, gt_series)):
                try:
                    # Clean and normalize the date strings
                    ocr_date = str(ocr_date).strip()
                    gt_date = str(gt_date).strip()
                    
                    # Try to parse dates
                    ocr_dt = datetime.strptime(ocr_date, rules['format'])
                    gt_dt = datetime.strptime(gt_date, rules['format'])
                    
                    # Compare dates
                    if ocr_dt == gt_dt:
                        results['correct_entries'] += 1
                    else:
                        results['format_errors'].append({
                            'index': idx,
                            'ocr_value': ocr_date,
                            'gt_value': gt_date,
                            'error': f'Date mismatch: OCR={ocr_date}, GT={gt_date}'
                        })
                        
                except ValueError as e:
                    results['format_errors'].append({
                        'index': idx,
                        'value': ocr_date,
                        'error': f'Invalid date format: {str(e)}'
                    })
            
            results['accuracy'] = self.calculate_accuracy(results['correct_entries'], results['total_entries'])
            return results
        except Exception as e:
            raise Exception(f"Error validating date column: {str(e)}")

    def validate_disease_column(self, ocr_series: pd.Series, gt_series: pd.Series) -> Dict[str, Any]:
        """Validate disease names and categories."""
        try:
            results = {
                'category_errors': [],
                'spelling_errors': [],
                'total_entries': len(ocr_series),
                'correct_entries': 0,
                'accuracy': 0.0,
                'ocr_values': ocr_series.tolist(),
                'gt_values': gt_series.tolist()
            }
            
            rules = self.get_validation_rules('disease')
            allowed_categories = set(cat.lower() for cat in rules['allowed_categories'])
            case_sensitive = rules.get('case_sensitive', False)
            
            for idx, (ocr_disease, gt_disease) in enumerate(zip(ocr_series, gt_series)):
                # Clean and normalize the disease strings
                ocr_disease = str(ocr_disease).strip()
                gt_disease = str(gt_disease).strip()
                
                if not case_sensitive:
                    ocr_disease = ocr_disease.lower()
                    gt_disease = gt_disease.lower()
                
                # Check if ground truth is in allowed categories
                if gt_disease not in allowed_categories:
                    results['category_errors'].append({
                        'index': idx,
                        'value': gt_disease,
                        'error': f'Disease not in allowed categories: {gt_disease}'
                    })
                    continue
                
                # Compare diseases
                if ocr_disease == gt_disease:
                    results['correct_entries'] += 1
                else:
                    results['spelling_errors'].append({
                        'index': idx,
                        'ocr_value': ocr_disease,
                        'gt_value': gt_disease,
                        'error': f'Disease mismatch: OCR={ocr_disease}, GT={gt_disease}'
                    })
            
            results['accuracy'] = self.calculate_accuracy(results['correct_entries'], results['total_entries'])
            return results
        except Exception as e:
            raise Exception(f"Error validating disease column: {str(e)}")

    def validate_panel_column(self, ocr_series: pd.Series, gt_series: pd.Series) -> Dict[str, Any]:
        """Validate panel identifiers."""
        try:
            results = {
                'format_errors': [],
                'range_errors': [],
                'sequence_errors': [],
                'total_entries': len(ocr_series),
                'correct_entries': 0,
                'accuracy': 0.0,
                'ocr_values': ocr_series.tolist(),
                'gt_values': gt_series.tolist()
            }
            
            rules = self.get_validation_rules('panel')
            min_value = rules['min_value']
            max_value = rules['max_value']
            
            for idx, (ocr_panel, gt_panel) in enumerate(zip(ocr_series, gt_series)):
                try:
                    # Convert to integers
                    ocr_panel = int(str(ocr_panel).strip())
                    gt_panel = int(str(gt_panel).strip())
                    
                    # Check if ground truth is in valid range
                    if not (min_value <= gt_panel <= max_value):
                        results['range_errors'].append({
                            'index': idx,
                            'value': gt_panel,
                            'error': f'Ground truth panel number {gt_panel} is not in valid range [{min_value}-{max_value}]'
                        })
                        continue
                    
                    # Check if OCR value is in valid range
                    if not (min_value <= ocr_panel <= max_value):
                        results['range_errors'].append({
                            'index': idx,
                            'value': ocr_panel,
                            'error': f'OCR panel number {ocr_panel} is not in valid range [{min_value}-{max_value}]'
                        })
                        continue
                    
                    if ocr_panel == gt_panel:
                        results['correct_entries'] += 1
                    else:
                        results['sequence_errors'].append({
                            'index': idx,
                            'ocr_value': ocr_panel,
                            'gt_value': gt_panel,
                            'error': 'Panel number mismatch'
                        })
                        
                except ValueError:
                    results['format_errors'].append({
                        'index': idx,
                        'value': ocr_panel,
                        'error': 'Invalid panel number format (must be an integer)'
                    })
            
            results['accuracy'] = self.calculate_accuracy(results['correct_entries'], results['total_entries'])
            return results
        except Exception as e:
            raise Exception(f"Error validating panel column: {str(e)}")

    def validate_row_column(self, ocr_series: pd.Series, gt_series: pd.Series) -> Dict[str, Any]:
        """Validate row numbers."""
        try:
            results = {
                'format_errors': [],
                'sequence_errors': [],
                'total_entries': len(ocr_series),
                'correct_entries': 0,
                'accuracy': 0.0,
                'ocr_values': ocr_series.tolist(),
                'gt_values': gt_series.tolist()
            }
            
            for idx, (ocr_row, gt_row) in enumerate(zip(ocr_series, gt_series)):
                try:
                    ocr_row = int(str(ocr_row).strip())
                    gt_row = int(str(gt_row).strip())
                    
                    if ocr_row == gt_row:
                        results['correct_entries'] += 1
                    else:
                        results['sequence_errors'].append({
                            'index': idx,
                            'ocr_value': ocr_row,
                            'gt_value': gt_row,
                            'error': 'Row number mismatch'
                        })
                        
                except ValueError:
                    results['format_errors'].append({
                        'index': idx,
                        'value': ocr_row,
                        'error': 'Invalid row number format'
                    })
            
            results['accuracy'] = self.calculate_accuracy(results['correct_entries'], results['total_entries'])
            return results
        except Exception as e:
            raise Exception(f"Error validating row column: {str(e)}")

    def validate_column(self, ocr_series: pd.Series, gt_series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Validate a specific column based on its type.
        
        Args:
            ocr_series: Series containing OCR values
            gt_series: Series containing ground truth values
            column_name: Name of the column being validated
            
        Returns:
            Dictionary containing validation results
            
        Raises:
            ValueError: If column type is unknown
        """
        if column_name == 'date':
            return self.validate_date_column(ocr_series, gt_series)
        elif column_name == 'disease':
            return self.validate_disease_column(ocr_series, gt_series)
        elif column_name == 'panel':
            return self.validate_panel_column(ocr_series, gt_series)
        elif column_name == 'row':
            return self.validate_row_column(ocr_series, gt_series)
        else:
            raise ValueError(f"Unknown column type: {column_name}")

    def check_index_columns(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, Any]:
        """Check if index columns from ground truth appear in OCR data.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            
        Returns:
            Dictionary containing presence check results
        """
        try:
            results = {}
            
            # Get all columns from both dataframes
            ocr_cols = ocr_df.columns.tolist()
            gt_cols = gt_df.columns.tolist()
            
            # Check each index column
            for col, idx in self.column_indices.items():
                col_info = {
                    'index': idx,
                    'ocr_col_name': ocr_cols[idx] if idx < len(ocr_cols) else None,
                    'gt_col_name': gt_cols[idx] if idx < len(gt_cols) else None,
                    'ocr_values': [],
                    'gt_values': []
                }
                
                # Get values from both dataframes using index
                if idx < len(ocr_cols):
                    col_info['ocr_values'] = ocr_df.iloc[:, idx].tolist()
                if idx < len(gt_cols):
                    col_info['gt_values'] = gt_df.iloc[:, idx].tolist()
                    
                results[col] = col_info
                
            return results
        except Exception as e:
            raise Exception(f"Error checking index columns: {str(e)}")

    def analyze_columns(self, ocr_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze index columns separately from data columns.
        
        Args:
            ocr_df: DataFrame containing OCR results
            gt_df: DataFrame containing ground truth data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # First check for column presence
            presence_check = self.check_index_columns(ocr_df, gt_df)
            
            # Then run detailed validation
            validation_results = {}
            for col in self.columns:
                if col not in ocr_df.columns or col not in gt_df.columns:
                    validation_results[col] = {
                        'error': f'Missing column: {col}',
                        'status': 'error',
                        'index': self.column_indices.get(col, 'unknown')
                    }
                    continue
                    
                # Validate each column type
                col_results = self.validate_column(ocr_df[col], gt_df[col], col)
                col_results['index'] = self.column_indices.get(col, 'unknown')
                validation_results[col] = col_results
            
            return {
                'presence_check': presence_check,
                'validation_results': validation_results
            }
        except Exception as e:
            raise Exception(f"Error analyzing columns: {str(e)}") 
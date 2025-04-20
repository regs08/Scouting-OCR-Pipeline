import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
from pathlib import Path

class ConfusionMatrixCalculator:
    """
    Calculator for confusion matrices with focus on text comparison.
    """
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize the confusion matrix calculator.
        
        Args:
            case_sensitive: Whether to perform case-sensitive comparisons
        """
        self.case_sensitive = case_sensitive
        
    def binary_confusion_matrix(
        self, 
        true_labels: pd.Series, 
        pred_labels: pd.Series, 
        positive_class: str
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate a binary confusion matrix for one-vs-all classification.
        
        Args:
            true_labels: Series of true labels
            pred_labels: Series of predicted labels
            positive_class: The value to consider as the positive class
            
        Returns:
            Dictionary containing confusion matrix metrics (TP, FP, FN, TN, precision, recall, F1)
        """
        # Convert to same case if not case sensitive
        if not self.case_sensitive:
            true_labels = true_labels.str.lower()
            pred_labels = pred_labels.str.lower()
            positive_class = positive_class.lower()
        
        # Create binary vectors
        true_binary = (true_labels == positive_class)
        pred_binary = (pred_labels == positive_class)
        
        # Calculate confusion matrix elements
        tp = sum(true_binary & pred_binary)
        fp = sum(~true_binary & pred_binary)
        fn = sum(true_binary & ~pred_binary)
        tn = sum(~true_binary & ~pred_binary)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def multiclass_confusion_matrix(
        self, 
        true_values: Union[pd.Series, List, np.ndarray], 
        pred_values: Union[pd.Series, List, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calculate a multiclass confusion matrix.
        
        Args:
            true_values: Array-like of true labels/values
            pred_values: Array-like of predicted labels/values
            
        Returns:
            DataFrame containing the confusion matrix where rows=true values, columns=predicted values
        """
        # Convert inputs to pandas Series for consistency
        true_series = pd.Series(true_values)
        pred_series = pd.Series(pred_values)
        
        # Apply case conversion if needed
        if not self.case_sensitive:
            true_series = true_series.astype(str).str.lower()
            pred_series = pred_series.astype(str).str.lower()
        
        # Get unique values from both series
        all_values = sorted(set(true_series) | set(pred_series))
        
        # Create confusion matrix as a defaultdict
        confusion_dict = defaultdict(lambda: defaultdict(int))
        
        # Fill the confusion matrix
        for true_val, pred_val in zip(true_series, pred_series):
            confusion_dict[true_val][pred_val] += 1
        
        # Convert to DataFrame
        confusion_df = pd.DataFrame(confusion_dict).fillna(0)
        
        # Ensure all values are represented in both rows and columns
        for val in all_values:
            if val not in confusion_df.columns:
                confusion_df[val] = 0
            if val not in confusion_df.index:
                confusion_df.loc[val] = 0
        
        # Sort index and columns for consistency
        confusion_df = confusion_df.reindex(sorted(confusion_df.columns), axis=1)
        confusion_df = confusion_df.reindex(sorted(confusion_df.index), axis=0)
        
        return confusion_df
    
    def dataframe_confusion_matrix(
        self, 
        gt_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate a combined confusion matrix for multiple columns in DataFrames.
        
        Args:
            gt_df: DataFrame containing ground truth data
            pred_df: DataFrame containing prediction data
            columns: List of columns to analyze (if None, use all common columns)
            
        Returns:
            DataFrame containing the combined confusion matrix
        """
        # Determine columns to use
        if columns is None:
            columns = list(set(gt_df.columns) & set(pred_df.columns))
            
        if not columns:
            raise ValueError("No common columns found between DataFrames")
        
        # Reset index for clean access
        pred_df = pred_df[columns].reset_index(drop=True).fillna("")
        gt_df = gt_df[columns].reset_index(drop=True).fillna("")
        
        # Initialize confusion matrix dictionary
        confusion_dict = defaultdict(lambda: defaultdict(int))
        
        # Count occurrences of each value pair across all specified columns
        for col in columns:
            true_vals = gt_df[col].astype(str)
            pred_vals = pred_df[col].astype(str)
            
            if not self.case_sensitive:
                true_vals = true_vals.str.lower()
                pred_vals = pred_vals.str.lower()
            
            # Update confusion matrix
            for true_val, pred_val in zip(true_vals, pred_vals):
                confusion_dict[true_val][pred_val] += 1
        
        # Convert to DataFrame
        confusion_df = pd.DataFrame.from_dict(confusion_dict, orient='index').fillna(0)
        
        # Sort index and columns
        confusion_df = confusion_df.sort_index()
        confusion_df = confusion_df.reindex(sorted(confusion_df.columns), axis=1)
        
        return confusion_df
    
    def get_confusion_metrics(self, confusion_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate metrics from a confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix DataFrame
            
        Returns:
            Dictionary with accuracy, macro precision, macro recall, and macro F1
        """
        # Calculate overall accuracy
        total = confusion_matrix.sum().sum()
        correct = sum(confusion_matrix.loc[val, val] for val in confusion_matrix.index 
                     if val in confusion_matrix.columns)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-class metrics
        class_metrics = {}
        for cls in confusion_matrix.index:
            if cls not in confusion_matrix.columns:
                continue
                
            true_pos = confusion_matrix.loc[cls, cls]
            false_pos = confusion_matrix[cls].sum() - true_pos
            false_neg = confusion_matrix.loc[cls].sum() - true_pos
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Calculate macro averages
        if class_metrics:
            macro_precision = sum(m['precision'] for m in class_metrics.values()) / len(class_metrics)
            macro_recall = sum(m['recall'] for m in class_metrics.values()) / len(class_metrics)
            macro_f1 = sum(m['f1_score'] for m in class_metrics.values()) / len(class_metrics)
        else:
            macro_precision = macro_recall = macro_f1 = 0
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics
        }
    
    def combine_confusion_matrices(
        self, 
        matrices: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine multiple confusion matrices into one.
        
        Args:
            matrices: List of confusion matrix DataFrames
            
        Returns:
            Combined confusion matrix DataFrame
        """
        if not matrices:
            return pd.DataFrame()
            
        # Get all unique indices and columns
        all_indices = set()
        all_columns = set()
        
        for matrix in matrices:
            all_indices.update(matrix.index)
            all_columns.update(matrix.columns)
        
        # Create combined matrix with all indices and columns
        combined = pd.DataFrame(0, index=sorted(all_indices), columns=sorted(all_columns))
        
        # Add each matrix to the combined matrix
        for matrix in matrices:
            for idx in matrix.index:
                for col in matrix.columns:
                    combined.loc[idx, col] += matrix.loc[idx, col]
        
        return combined 
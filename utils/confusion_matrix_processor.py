import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
from pathlib import Path
from utils.base_processor import BaseProcessor

class ConfusionMatrixProcessor(BaseProcessor):
    """
    Processor for creating, analyzing, and visualizing confusion matrices for OCR data.
    This class supports one-vs-all confusion matrices for classification tasks.
    """
    
    def __init__(
        self, 
        case_sensitive: bool = False,
        output_dir: Optional[str] = None,
        verbose: bool = False,
        enable_logging: bool = False
    ):
        """
        Initialize the ConfusionMatrixProcessor.
        
        Args:
            case_sensitive: Whether to perform case-sensitive comparisons
            output_dir: Directory to save confusion matrix outputs
            verbose: Whether to display detailed output
            enable_logging: Whether to enable logging
        """
        super().__init__(verbose=verbose, enable_logging=enable_logging)
        self.case_sensitive = case_sensitive
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output", "confusion_matrices")
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def create_binary_confusion_matrix(
        self, 
        true_labels: pd.Series, 
        pred_labels: pd.Series, 
        positive_class: str
    ) -> Dict[str, Union[int, float]]:
        """
        Create a binary confusion matrix for one-vs-all classification.
        
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
    
    def create_multiclass_confusion_matrix(
        self, 
        gt_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        columns: List[str],
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create a multiclass confusion matrix for the comparison.
        
        Args:
            gt_df: DataFrame containing ground truth data
            pred_df: DataFrame containing prediction data
            columns: List of columns to analyze
            max_rows: Maximum number of rows to analyze
            
        Returns:
            DataFrame containing the confusion matrix
        """
        try:
            self.display_and_log("Creating multiclass confusion matrix...", {
                "Columns": columns,
                "Max Rows": max_rows or "All"
            })

            # Reset index for clean access
            pred_df = pred_df[columns].reset_index(drop=True).fillna("").astype(str)
            gt_df = gt_df[columns].reset_index(drop=True).fillna("").astype(str)

            row_limit = max_rows or min(len(pred_df), len(gt_df))
            
            # Initialize confusion matrix dictionary
            confusion_dict = defaultdict(lambda: defaultdict(int))
            
            # Count occurrences of each value pair
            for i in range(row_limit):
                for col in columns:
                    val_pred = pred_df.at[i, col]
                    val_gt = gt_df.at[i, col]
                    if not self.case_sensitive:
                        val_pred = val_pred.lower()
                        val_gt = val_gt.lower()
                    confusion_dict[val_gt][val_pred] += 1
            
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
            
            self.display_and_log("Multiclass confusion matrix created", {
                "Total Predictions": total_predictions,
                "Correct Predictions": correct_predictions,
                "Accuracy": f"{accuracy:.2f}%"
            })
            
            return confusion_df
        except Exception as e:
            error_msg = f"Error creating multiclass confusion matrix: {str(e)}"
            self.display_and_log(error_msg)
            raise Exception(error_msg)
    
    def analyze_confusion_patterns(
        self, 
        gt_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        columns: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze what true values are being incorrectly predicted as.
        
        Args:
            gt_df: DataFrame containing ground truth data
            pred_df: DataFrame containing prediction data
            columns: List of columns to analyze
            
        Returns:
            Dictionary mapping true values to their incorrect predictions
        """
        confusion_patterns = defaultdict(lambda: defaultdict(int))
        
        for col in columns:
            gt_series = gt_df[col].astype(str)
            pred_series = pred_df[col].astype(str)
            
            if not self.case_sensitive:
                gt_series = gt_series.str.lower()
                pred_series = pred_series.str.lower()
            
            # Find mismatches and count their occurrences
            mismatches = gt_series != pred_series
            for gt_val, pred_val in zip(gt_series[mismatches], pred_series[mismatches]):
                confusion_patterns[gt_val][pred_val] += 1
        
        return dict(confusion_patterns)
    
    def analyze_reverse_confusion_patterns(
        self, 
        gt_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        columns: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze what prediction values are incorrectly predicted from true values.
        
        Args:
            gt_df: DataFrame containing ground truth data
            pred_df: DataFrame containing prediction data
            columns: List of columns to analyze
            
        Returns:
            Dictionary mapping predicted values to the true values they came from
        """
        reverse_confusion_patterns = defaultdict(lambda: defaultdict(int))
        
        for col in columns:
            gt_series = gt_df[col].astype(str)
            pred_series = pred_df[col].astype(str)
            
            if not self.case_sensitive:
                gt_series = gt_series.str.lower()
                pred_series = pred_series.str.lower()
            
            # Find mismatches and count their occurrences
            mismatches = gt_series != pred_series
            for gt_val, pred_val in zip(gt_series[mismatches], pred_series[mismatches]):
                # Store as pred_val -> gt_val mapping
                reverse_confusion_patterns[pred_val][gt_val] += 1
        
        return dict(reverse_confusion_patterns)
    
    def one_vs_all_analysis(
        self, 
        gt_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        columns: List[str],
        unique_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a one-vs-all analysis for each unique value in the datasets.
        
        Args:
            gt_df: DataFrame containing ground truth data
            pred_df: DataFrame containing prediction data
            columns: List of columns to analyze
            unique_id: Unique identifier for the analysis (used for saving results)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Get all unique values from both ground truth and predictions
            gt_values = set()
            pred_values = set()
            
            for col in gt_df.columns:
                if not self.case_sensitive:
                    gt_values.update(gt_df[col].astype(str).str.lower().unique())
                    pred_values.update(pred_df[col].astype(str).str.lower().unique())
                else:
                    gt_values.update(gt_df[col].astype(str).unique())
                    pred_values.update(pred_df[col].astype(str).unique())
            
            all_values = sorted(gt_values.union(pred_values))
            
            # Create binary confusion matrices for each unique value
            value_metrics = {}
            for value in all_values:
                metrics = {}
                for col in columns:
                    true_labels = gt_df[col].astype(str)
                    pred_labels = pred_df[col].astype(str)
                    
                    col_metrics = self.create_binary_confusion_matrix(true_labels, pred_labels, value)
                    metrics[col] = col_metrics
                
                # Aggregate metrics across all columns
                total_metrics = {
                    'true_positives': sum(m['true_positives'] for m in metrics.values()),
                    'false_positives': sum(m['false_positives'] for m in metrics.values()),
                    'false_negatives': sum(m['false_negatives'] for m in metrics.values()),
                    'true_negatives': sum(m['true_negatives'] for m in metrics.values())
                }
                
                # Calculate overall precision, recall, and F1
                tp = total_metrics['true_positives']
                fp = total_metrics['false_positives']
                fn = total_metrics['false_negatives']
                
                total_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                total_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                total_metrics['f1_score'] = 2 * (total_metrics['precision'] * total_metrics['recall']) / \
                    (total_metrics['precision'] + total_metrics['recall']) \
                    if (total_metrics['precision'] + total_metrics['recall']) > 0 else 0
                
                value_metrics[value] = total_metrics
            
            # Analyze confusion patterns
            confusion_patterns = self.analyze_confusion_patterns(gt_df, pred_df, columns)
            reverse_patterns = self.analyze_reverse_confusion_patterns(gt_df, pred_df, columns)
            
            # Create summary DataFrame
            summary_data = []
            for value, metrics in value_metrics.items():
                # Get the top confused values for this ground truth value
                confused_with = sorted(
                    confusion_patterns.get(value, {}).items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                # Get the top ground truth values that were predicted as this value
                predicted_from = sorted(
                    reverse_patterns.get(value, {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                confused_str = "; ".join([
                    f"'{pred_val}' ({count} times)" 
                    for pred_val, count in confused_with
                ]) if confused_with else "No confusion"
                
                predicted_from_str = "; ".join([
                    f"'{gt_val}' ({count} times)"
                    for gt_val, count in predicted_from
                ]) if predicted_from else "No incorrect predictions"
                
                summary_data.append({
                    'Value': value,
                    'True Positives': metrics['true_positives'],
                    'False Positives': metrics['false_positives'],
                    'False Negatives': metrics['false_negatives'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1_score'],
                    'Predicted Instead Of': confused_str,
                    'Incorrectly Predicted From': predicted_from_str
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('F1 Score', ascending=False)
            
            # Format percentage columns
            for col in ['Precision', 'Recall', 'F1 Score']:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2%}")
            
            # Save to CSV if unique_id is provided
            if unique_id:
                output_path = os.path.join(self.output_dir, f"{unique_id}_one_vs_all_analysis.csv")
                summary_df.to_csv(output_path, index=False)
                self.display_and_log(f"Analysis saved to {output_path}")
            
            results = {
                'summary': summary_df,
                'detailed_metrics': value_metrics,
                'confusion_patterns': confusion_patterns,
                'reverse_patterns': reverse_patterns
            }
            
            return results
            
        except Exception as e:
            error_msg = f"Error in one-vs-all analysis: {str(e)}"
            self.display_and_log(error_msg)
            raise Exception(error_msg)
    
    def aggregate_results(
        self, 
        analysis_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Aggregate results across multiple analyses.
        
        Args:
            analysis_results: Dictionary mapping unique IDs to analysis results
            
        Returns:
            DataFrame containing aggregated metrics
        """
        try:
            # Get all unique values across all analyses
            all_values = set()
            for result in analysis_results.values():
                all_values.update(result['detailed_metrics'].keys())
            
            # Combine metrics across all files
            overall_metrics = {}
            for value in all_values:
                total_tp = sum(results['detailed_metrics'].get(value, {}).get('true_positives', 0) 
                             for results in analysis_results.values())
                total_fp = sum(results['detailed_metrics'].get(value, {}).get('false_positives', 0) 
                             for results in analysis_results.values())
                total_fn = sum(results['detailed_metrics'].get(value, {}).get('false_negatives', 0) 
                             for results in analysis_results.values())
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                overall_metrics[value] = {
                    'Value': value,
                    'True Positives': total_tp,
                    'False Positives': total_fp,
                    'False Negatives': total_fn,
                    'Precision': f"{precision:.2%}",
                    'Recall': f"{recall:.2%}",
                    'F1 Score': f"{f1:.2%}"
                }
            
            # Create DataFrame
            overall_df = pd.DataFrame(overall_metrics.values())
            overall_df = overall_df.sort_values('F1 Score', ascending=False)
            
            # Save overall results
            output_path = os.path.join(self.output_dir, "overall_one_vs_all_analysis.csv")
            overall_df.to_csv(output_path, index=False)
            self.display_and_log(f"Aggregated results saved to {output_path}")
            
            return overall_df
            
        except Exception as e:
            error_msg = f"Error in aggregate_results: {str(e)}"
            self.display_and_log(error_msg)
            raise Exception(error_msg)
    
    def visualize_confusion_matrix(
        self,
        confusion_matrix: pd.DataFrame,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize a confusion matrix.
        
        Args:
            confusion_matrix: DataFrame containing the confusion matrix
            title: Title for the plot
            figsize: Figure size (width, height)
            save_path: Path to save the visualization (if None, will not save)
        """
        try:
            # This requires matplotlib, so we import it here in case it's not available
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Limit the size of the matrix for better visualization
            # If matrix is very large, take the top N values by frequency
            MAX_MATRIX_SIZE = 30
            if confusion_matrix.shape[0] > MAX_MATRIX_SIZE or confusion_matrix.shape[1] > MAX_MATRIX_SIZE:
                # Get row and column sums
                row_sums = confusion_matrix.sum(axis=1)
                col_sums = confusion_matrix.sum(axis=0)
                
                # Keep top values by frequency
                top_rows = row_sums.nlargest(MAX_MATRIX_SIZE).index
                top_cols = col_sums.nlargest(MAX_MATRIX_SIZE).index
                
                # Filter the matrix
                filtered_matrix = confusion_matrix.loc[top_rows, top_cols]
                self.display_and_log(f"Matrix too large, showing top {MAX_MATRIX_SIZE} values by frequency")
            else:
                filtered_matrix = confusion_matrix
            
            # Adjust figure size based on matrix dimensions
            matrix_size = max(filtered_matrix.shape)
            scale_factor = max(1, matrix_size / 10)  # Scale based on matrix size
            adjusted_figsize = (figsize[0] * scale_factor, figsize[1] * scale_factor)
            
            plt.figure(figsize=adjusted_figsize)
            
            # Create heatmap with customized settings for better readability
            # Use smaller font size for larger matrices
            font_size = max(6, 12 - (matrix_size // 10))
            
            # Format annotation to reduce clutter and show scientific notation for large numbers
            fmt = '.0f'
            if filtered_matrix.values.max() > 1000:
                fmt = '.1e'
            
            ax = sns.heatmap(
                filtered_matrix, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                linewidths=0.5,
                annot_kws={"size": font_size},
                cbar_kws={"shrink": 0.7}  # Make colorbar smaller
            )
            
            # Rotate labels for better readability
            plt.title(title, fontsize=12 + (4 / scale_factor))
            plt.ylabel('True Values', fontsize=10 + (2 / scale_factor))
            plt.xlabel('Predicted Values', fontsize=10 + (2 / scale_factor))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right', fontsize=font_size)
            plt.yticks(rotation=0, fontsize=font_size)
            
            # Tight layout to maximize use of figure area
            plt.tight_layout()
            
            if save_path:
                # Use higher DPI for better quality
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                self.display_and_log(f"Visualization saved to {save_path}")
            
            plt.close()
        except ImportError:
            self.display_and_log("Visualization requires matplotlib and seaborn packages")
        except Exception as e:
            error_msg = f"Error in visualize_confusion_matrix: {str(e)}"
            self.display_and_log(error_msg) 
            
    def aggregate_confusion_matrices(
        self,
        confusion_matrices: List[pd.DataFrame],
        title: str = "Aggregated Confusion Matrix",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        max_display_size: int = 40
    ) -> pd.DataFrame:
        """
        Aggregate multiple confusion matrices into one and visualize it.
        
        Args:
            confusion_matrices: List of confusion matrix DataFrames to aggregate
            title: Title for the visualization
            save_path: Path to save the visualization
            figsize: Base figure size for the visualization
            max_display_size: Maximum number of rows/columns to display
            
        Returns:
            The aggregated confusion matrix DataFrame
        """
        try:
            self.display_and_log(f"Aggregating {len(confusion_matrices)} confusion matrices")
            
            if not confusion_matrices:
                raise ValueError("No confusion matrices provided for aggregation")
                
            # Get all unique row and column indices
            all_rows = set()
            all_cols = set()
            for matrix in confusion_matrices:
                all_rows.update(matrix.index)
                all_cols.update(matrix.columns)
            
            # Sort indices for consistency
            all_rows = sorted(all_rows)
            all_cols = sorted(all_cols)
            
            # Create a new DataFrame with all possible indices
            aggregated_matrix = pd.DataFrame(0, index=all_rows, columns=all_cols)
            
            # Sum up all matrices
            for matrix in confusion_matrices:
                for row in matrix.index:
                    for col in matrix.columns:
                        aggregated_matrix.at[row, col] += matrix.at[row, col]
            
            # Calculate totals for filtering
            row_totals = aggregated_matrix.sum(axis=1)
            col_totals = aggregated_matrix.sum(axis=0)
            
            # Get accuracy before filtering
            total_predictions = aggregated_matrix.sum().sum()
            correct_predictions = sum(aggregated_matrix[col][col] for col in aggregated_matrix.columns 
                                    if col in aggregated_matrix.index)
            accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            self.display_and_log("Aggregated confusion matrix created", {
                "Total Values": len(all_rows),
                "Total Predictions": total_predictions,
                "Correct Predictions": correct_predictions,
                "Overall Accuracy": f"{accuracy:.2f}%"
            })
            
            # Save the full matrix before filtering for visualization
            if save_path:
                full_save_path = save_path.replace('.png', '_full.csv')
                aggregated_matrix.to_csv(full_save_path)
                self.display_and_log(f"Full aggregated matrix saved to {full_save_path}")
            
            # Filter for visualization if necessary
            if len(all_rows) > max_display_size or len(all_cols) > max_display_size:
                # Get top rows and columns by frequency
                top_rows = row_totals.nlargest(max_display_size).index
                top_cols = col_totals.nlargest(max_display_size).index
                
                # Create a filtered view for visualization
                vis_matrix = aggregated_matrix.loc[top_rows, top_cols]
                
                # Calculate filtered view statistics
                filtered_total = vis_matrix.sum().sum()
                filtered_correct = sum(vis_matrix[col][col] for col in vis_matrix.columns 
                                    if col in vis_matrix.index)
                filtered_accuracy = (filtered_correct / filtered_total * 100) if filtered_total > 0 else 0
                
                self.display_and_log(f"Filtered matrix for visualization to top {max_display_size} values", {
                    "Filtered Total": filtered_total,
                    "Filtered Correct": filtered_correct,
                    "Filtered Accuracy": f"{filtered_accuracy:.2f}%",
                    "Percentage of Data Retained": f"{filtered_total / total_predictions * 100:.2f}%"
                })
                
                vis_title = f"{title} (Top {max_display_size} values, {filtered_accuracy:.1f}% accuracy)"
            else:
                vis_matrix = aggregated_matrix
                vis_title = f"{title} ({accuracy:.1f}% accuracy)"
            
            # Visualize the matrix
            if save_path:
                try:
                    self.visualize_confusion_matrix(
                        confusion_matrix=vis_matrix,
                        title=vis_title,
                        figsize=figsize,
                        save_path=save_path
                    )
                except Exception as vis_error:
                    self.display_and_log(f"Warning: Visualization failed: {str(vis_error)}")
            
            return aggregated_matrix
            
        except Exception as e:
            error_msg = f"Error in aggregate_confusion_matrices: {str(e)}"
            self.display_and_log(error_msg)
            raise Exception(error_msg) 
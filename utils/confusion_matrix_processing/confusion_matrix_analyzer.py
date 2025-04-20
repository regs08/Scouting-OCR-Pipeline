from typing import Dict, List, Optional, Union, Any, Tuple
from utils.confusion_matrix_processing.confusion_matrix_calculator import ConfusionMatrixCalculator
from utils.confusion_matrix_processing.confusion_matrix_visualizer import ConfusionMatrixVisualizer
from utils.confusion_matrix_processing.confusion_matrix_storage import ConfusionMatrixStorage
import pandas as pd
import numpy as np
from pathlib import Path

class ConfusionMatrixAnalyzer:
    """
    Class for analyzing confusion matrices and generating visualizations and reports.
    """
    
    def __init__(
        self, 
        output_dir: Optional[Union[str, Path]] = None,
        case_sensitive: bool = False
    ):
        """
        Initialize the analyzer with calculator, visualizer, and storage components.
        
        Args:
            output_dir: Base directory for outputs
            case_sensitive: Whether to perform case-sensitive comparisons
        """
        # Initialize base directory
        self.base_dir = Path(output_dir) if output_dir else Path.cwd() / "confusion_matrix_analysis"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.viz_dir = self.base_dir / "visualizations"
        self.data_dir = self.base_dir / "data"
        self.report_dir = self.base_dir / "reports"
        
        for directory in [self.viz_dir, self.data_dir, self.report_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.calculator = ConfusionMatrixCalculator(case_sensitive=case_sensitive)
        self.visualizer = ConfusionMatrixVisualizer(output_dir=self.viz_dir)
        self.storage = ConfusionMatrixStorage(output_dir=self.data_dir)
    
    def analyze_dataframes(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        identifier: str = "analysis",
        save_results: bool = True,
        create_visualizations: bool = True,
        aggregate_results: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze two DataFrames by creating confusion matrices and metrics.
        
        Args:
            gt_df: Ground truth DataFrame
            pred_df: Prediction DataFrame
            columns: Specific columns to analyze (if None, use all common columns)
            identifier: Unique identifier for this analysis
            save_results: Whether to save results to files
            create_visualizations: Whether to create visualizations
            aggregate_results: Whether to add matrix to aggregation
            
        Returns:
            Dictionary containing analysis results
        """
        # Create confusion matrix
        confusion_matrix = self.calculator.dataframe_confusion_matrix(
            gt_df=gt_df,
            pred_df=pred_df,
            columns=columns
        )
        
        # Calculate metrics
        metrics = self.calculator.get_confusion_metrics(confusion_matrix)
        
        # Prepare results dictionary
        results = {
            'identifier': identifier,
            'confusion_matrix': confusion_matrix,
            'metrics': metrics,
            'file_paths': {}
        }
        
        # Save results if requested
        if save_results:
            # Save confusion matrix
            matrix_path = self.storage.save_matrix(
                confusion_matrix=confusion_matrix,
                identifier=identifier
            )
            results['file_paths']['confusion_matrix'] = matrix_path
            
            # Save metrics
            metrics_paths = self.storage.save_metrics(
                metrics=metrics,
                identifier=identifier
            )
            results['file_paths'].update(metrics_paths)
        
        # Create visualizations if requested
        if create_visualizations:
            # Visualize confusion matrix
            matrix_viz_path = self.visualizer.visualize(
                confusion_matrix=confusion_matrix,
                title=f"Confusion Matrix - {identifier}"
            )
            results['file_paths']['matrix_visualization'] = matrix_viz_path
            
            # Visualize class metrics
            if 'class_metrics' in metrics:
                metrics_viz_path = self.visualizer.visualize_class_metrics(
                    metrics=metrics['class_metrics'],
                    title=f"Class Performance - {identifier}"
                )
                results['file_paths']['metrics_visualization'] = metrics_viz_path
        
        # Track this matrix for later aggregation if requested
        if aggregate_results:
            self._track_for_aggregation(matrix_path)
        
        return results
    
    def _track_for_aggregation(self, matrix_path: Union[str, Path]) -> None:
        """Track matrix file for later aggregation."""
        if not hasattr(self, '_matrices_to_aggregate'):
            self._matrices_to_aggregate = []
        self._matrices_to_aggregate.append(matrix_path)

    def finalize_analysis(self, identifier: str = "dataset_overview") -> Dict[str, Any]:
        """
        Finalize analysis by aggregating all processed matrices.
        Should be called after all individual analyses are complete.
        
        Args:
            identifier: Identifier for the aggregated results
            
        Returns:
            Dictionary containing aggregated results and file paths
        """
        if not hasattr(self, '_matrices_to_aggregate') or not self._matrices_to_aggregate:
            return {
                "status": "warning",
                "message": "No matrices to aggregate"
            }
        
        try:
            # Initialize aggregator
            from .confusion_matrix_aggregator import ConfusionMatrixAggregator
            aggregator = ConfusionMatrixAggregator(self.storage)
            
            # Aggregate all matrices
            aggregated_results = aggregator.aggregate_matrices(
                matrix_files=self._matrices_to_aggregate,
                output_identifier=identifier
            )
            
            if aggregated_results["status"] == "success":
                aggregated_matrix = aggregated_results["aggregated_matrix"]
                
                # Explicitly save the aggregated matrix
                matrix_save_path = self.storage.save_matrix(
                    confusion_matrix=aggregated_matrix,
                    identifier=f"{identifier}_aggregated",
                    format='csv'  # Explicitly specify format
                )
                
                # Calculate metrics for aggregated matrix
                aggregated_metrics = self.calculator.get_confusion_metrics(aggregated_matrix)
                
                # Save aggregated metrics
                metrics_paths = self.storage.save_metrics(
                    metrics=aggregated_metrics,
                    identifier=f"{identifier}_aggregated"
                )
                
                # Create visualizations with explicit paths
                viz_paths = {}
                
                # Main confusion matrix visualization
                matrix_viz_path = str(self.viz_dir / f"{identifier}_aggregated_matrix.png")
                self.visualizer.visualize(
                    confusion_matrix=aggregated_matrix,
                    title=f"Dataset Overview - Aggregated Confusion Matrix\n({aggregated_results['matrices_combined']} matrices combined)",
                    save_path=matrix_viz_path,
                    show=False  # Don't show, just save
                )
                viz_paths['matrix'] = matrix_viz_path
                
                # Class metrics visualization if available
                if 'class_metrics' in aggregated_metrics:
                    metrics_viz_path = str(self.viz_dir / f"{identifier}_aggregated_metrics.png")
                    self.visualizer.visualize_class_metrics(
                        metrics=aggregated_metrics['class_metrics'],
                        title=f"Dataset Overview - Aggregated Class Performance",
                        save_path=metrics_viz_path,
                        show=False  # Don't show, just save
                    )
                    viz_paths['class_metrics'] = metrics_viz_path
                
                # Clear tracked matrices
                self._matrices_to_aggregate = []
                
                # Print confirmation of saved files
                print(f"\nSaved aggregated files:")
                print(f"Matrix CSV: {matrix_save_path}")
                print(f"Visualization: {matrix_viz_path}")
                
                return {
                    "status": "success",
                    "matrices_combined": aggregated_results["matrices_combined"],
                    "total_predictions": aggregated_results["total_predictions"],
                    "aggregated_matrix": aggregated_matrix,
                    "metrics": aggregated_metrics,
                    "file_paths": {
                        "matrix_csv": matrix_save_path,
                        "metrics": metrics_paths,
                        "visualizations": viz_paths
                    }
                }
            
            return aggregated_results
            
        except Exception as e:
            print(f"Error during aggregation: {str(e)}")  # Print error for debugging
            return {
                "status": "error",
                "message": f"Error during aggregation: {str(e)}"
            }
    
    def batch_analyze(
        self,
        gt_dfs: Dict[str, pd.DataFrame],
        pred_dfs: Dict[str, pd.DataFrame],
        columns_map: Dict[str, List[str]] = None,
        save_results: bool = True,
        create_visualizations: bool = True,
        generate_reports: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple pairs of DataFrames in batch mode.
        
        Args:
            gt_dfs: Dictionary mapping identifiers to ground truth DataFrames
            pred_dfs: Dictionary mapping identifiers to prediction DataFrames
            columns_map: Dictionary mapping identifiers to columns to analyze
            save_results: Whether to save results to files
            create_visualizations: Whether to create visualizations
            
        Returns:
            Dictionary mapping identifiers to their analysis results
        """
        results = {}
        
        # Process each pair of DataFrames
        for identifier in gt_dfs.keys():
            if identifier not in pred_dfs:
                print(f"Warning: No prediction DataFrame found for {identifier}")
                continue
                
            gt_df = gt_dfs[identifier]
            pred_df = pred_dfs[identifier]
            
            # Get columns for this identifier if provided
            columns = columns_map.get(identifier) if columns_map else None
            
            # Run analysis
            analysis_result = self.analyze_dataframes(
                gt_df=gt_df,
                pred_df=pred_df,
                columns=columns,
                identifier=identifier,
                save_results=save_results,
                create_visualizations=create_visualizations
            )
            
            results[identifier] = analysis_result
            
        return results

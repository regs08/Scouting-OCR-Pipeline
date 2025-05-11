from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from utils.pipeline_component import PipelineComponent
from utils.data_classes import (
    MatchedFile,
    ConfusionPattern,
    ClassConfusion,
    ProblemClass,
    ConfusionAnalysis
)
from datetime import datetime

class ConfusionPatternAnalyzer:
    """Analyzes confusion patterns between classes, focusing on those with low F1 scores."""
    
    def __init__(self, top_n: int = 20, confusion_threshold: float = 0.01):
        """
        Initialize the analyzer.
        
        Args:
            top_n: Number of worst-performing classes to analyze
            confusion_threshold: Minimum confusion rate to consider (0-1)
        """
        self.top_n = top_n
        self.confusion_threshold = confusion_threshold
    
    def analyze_confusion_patterns(
        self,
        cm_df: pd.DataFrame,
        f1_scores: pd.Series
    ) -> ConfusionAnalysis:
        """
        Analyze confusion patterns for classes with lowest F1 scores.
        
        Args:
            cm_df: Confusion matrix DataFrame
            f1_scores: Series of F1 scores for each class
            
        Returns:
            ConfusionAnalysis object containing the analysis results
        """
        # Get problem classes (lowest F1 scores)
        # Sort by F1 score first, then by class name for consistency
        problem_classes = f1_scores.sort_values().nsmallest(self.top_n).index.tolist()
        
        # Initialize results
        problem_classes_dict = {}
        confusion_patterns = []
        
        # Analyze each problem class
        for class_a in problem_classes:
            class_confusions = {}
            
            # Get total predictions for this class
            total_a = cm_df.loc[class_a].sum()
            
            # Analyze confusions with other classes
            for class_b in sorted(cm_df.columns):  # Sort columns for consistency
                if class_b != class_a:
                    # Get confusion counts
                    a_to_b = cm_df.loc[class_a, class_b]  # A predicted as B
                    b_to_a = cm_df.loc[class_b, class_a]  # B predicted as A
                    
                    # Calculate rates
                    a_to_b_rate = a_to_b / total_a if total_a > 0 else 0
                    b_to_a_rate = b_to_a / cm_df.loc[class_b].sum() if cm_df.loc[class_b].sum() > 0 else 0
                    
                    # Only include if confusion rate is above threshold
                    if a_to_b_rate > self.confusion_threshold or b_to_a_rate > self.confusion_threshold:
                        class_confusion = ClassConfusion(
                            count=int(a_to_b),
                            rate=float(a_to_b_rate),
                            reverse_count=int(b_to_a),
                            reverse_rate=float(b_to_a_rate),
                            is_symmetric=abs(a_to_b_rate - b_to_a_rate) < 0.1
                        )
                        class_confusions[class_b] = class_confusion
                        
                        # Create confusion pattern
                        pattern = ConfusionPattern(
                            class_a=class_a,
                            class_b=class_b,
                            f1_a=float(f1_scores[class_a]),
                            f1_b=float(f1_scores[class_b]),
                            confusion_rate=float(a_to_b_rate),
                            reverse_rate=float(b_to_a_rate),
                            is_symmetric=class_confusion.is_symmetric
                        )
                        
                        # Check if reverse pattern exists
                        reverse_exists = False
                        for existing in confusion_patterns:
                            if (existing.class_a == class_b and existing.class_b == class_a):
                                reverse_exists = True
                                break
                        
                        if not reverse_exists:
                            confusion_patterns.append(pattern)
            
            # Add problem class
            problem_classes_dict[class_a] = ProblemClass(
                f1_score=float(f1_scores[class_a]),
                confused_with=class_confusions
            )
        
        # Sort confusion patterns by total confusion rate, then by class names for consistency
        confusion_patterns.sort(
            key=lambda x: (-(x.confusion_rate + x.reverse_rate), x.class_a, x.class_b)
        )
        
        return ConfusionAnalysis(
            problem_classes=problem_classes_dict,
            confusion_patterns={'most_common': confusion_patterns}
        )

class ConfusionMatrixComponent(PipelineComponent):
    """
    Component for generating confusion matrix analysis from matched files.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        enable_logging: bool = True,
        enable_console: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        operation_name: Optional[str] = None,
        top_n_problem_classes: int = 20,
        confusion_threshold: float = 0.01,
        include_columns: Optional[List[str]] = None,
        exclude_last_n_rows: int = 4,
        site_data: Optional[Any] = None,
        **kwargs: Any
    ):
        """
        Initialize the confusion matrix component.
        
        Args:
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
            top_n_problem_classes: Number of worst-performing classes to analyze
            confusion_threshold: Minimum confusion rate to consider (0-1)
            include_columns: List of column names to include in analysis. If None, uses L1-L20.
            exclude_last_n_rows: Number of rows to exclude from the end of each DataFrame (default: 4)
            site_data: Site data object (not used for column selection)
            **kwargs: Additional keyword arguments for component initialization
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "confusion_matrix",
            **kwargs
        )
        
        self.top_n_problem_classes = top_n_problem_classes
        self.confusion_threshold = confusion_threshold
        self.exclude_last_n_rows = exclude_last_n_rows
        
        # Set include_columns to L1-L20 if not specified
        if include_columns is None:
            self.include_columns = [f'L{i}' for i in range(1, 21)]
        else:
            self.include_columns = include_columns
            
        self.pattern_analyzer = ConfusionPatternAnalyzer(
            top_n=top_n_problem_classes,
            confusion_threshold=confusion_threshold
        )

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data before running the pipeline.
        Loads the CSV files and prepares them for confusion matrix computation.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dictionary with loaded dataframes and metadata
        """
        self.log_info("process_before_pipeline", "Loading CSV files for confusion matrix computation")
        
        # Get matched_files and path_manager from input data
        self.matched_files = input_data.get('matched_files', [])
        self.path_manager = input_data.get('path_manager')
        
        if not self.matched_files:
            self.log_warning("process_before_pipeline", "No matched files found in input data")
            raise ValueError("No matched files provided in input data")
            
        if not self.path_manager:
            self.log_warning("process_before_pipeline", "No path manager found in input data")
            raise ValueError("Path manager must be provided in input data")
            
        # Create checkpoint directory
        checkpoint_dir = self.path_manager.get_checkpoint_path(
            self.session_id,
            "confusion_matrix_analysis"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        loaded_files = {}
        for matched_file in self.matched_files:
            try:
                # Load the CSV files
                gt_df = pd.read_csv(matched_file.gt_path)
                pred_df = pd.read_csv(matched_file.pred_path)
                
                # Get common columns
                common_columns = list(set(gt_df.columns) & set(pred_df.columns))
                
                if not common_columns:
                    self.log_warning("process_before_pipeline", 
                        f"No common columns found between files: {matched_file.normalized_name}")
                    continue
                
                # Filter columns if specified
                if self.include_columns:
                    common_columns = [col for col in common_columns if col in self.include_columns]
                    if not common_columns:
                        self.log_warning("process_before_pipeline",
                            f"No matching columns found in include_columns for {matched_file.normalized_name}")
                        continue
                    
                    # Add debug logging for column selection
                    self.log_info("process_before_pipeline",
                        f"Selected columns for {matched_file.normalized_name}",
                        {
                            "selected_columns": common_columns,
                            "total_columns": len(common_columns)
                        }
                    )
                
                # Exclude last n rows if specified
                if self.exclude_last_n_rows > 0:
                    original_rows = len(gt_df)
                    gt_df = gt_df.iloc[:-self.exclude_last_n_rows]
                    pred_df = pred_df.iloc[:-self.exclude_last_n_rows]
                    
                    # Add debug logging for row exclusion
                    self.log_info("process_before_pipeline",
                        f"Row exclusion for {matched_file.normalized_name}",
                        {
                            "original_rows": original_rows,
                            "remaining_rows": len(gt_df),
                            "excluded_rows": self.exclude_last_n_rows
                        }
                    )
                
                loaded_files[matched_file.normalized_name] = {
                    'gt_df': gt_df,
                    'pred_df': pred_df,
                    'common_columns': common_columns
                }
                
                self.log_info("process_before_pipeline", 
                    f"Successfully loaded {matched_file.normalized_name}",
                    {
                        "columns": len(common_columns),
                        "rows": len(gt_df),
                        "excluded_rows": self.exclude_last_n_rows
                    }
                )
                
            except Exception as e:
                self.log_error("process_before_pipeline", 
                    f"Error loading {matched_file.normalized_name}: {str(e)}")
        
        return {
            **input_data,
            'loaded_files': loaded_files,
            'checkpoint_dir': checkpoint_dir,
            'matched_files': self.matched_files,
            'path_manager': self.path_manager
        }
    
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pipeline output data and prepare final results.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dictionary with processed results
        """
        # Get results from each component
        cm_results = pipeline_output.get('confusion_matrix_results', {})
        viz_results = pipeline_output.get('visualization_results', {})
        error_results = pipeline_output.get('error_analysis_results', {})
        
        # Create summary of results
        summary = {
            'total_files_analyzed': len(self.matched_files),
            'successful_analyses': sum(1 for r in cm_results.values() if r.get('status') == 'success'),
            'failed_analyses': sum(1 for r in cm_results.values() if r.get('status') == 'error'),
            'checkpoint_dir': pipeline_output.get('checkpoint_dir')
        }
        
        # Add aggregated results if available
        if 'aggregated' in cm_results:
            agg_results = cm_results['aggregated']
            if agg_results.get('status') == 'success':
                summary['aggregated_results'] = {
                    'confusion_matrix': agg_results.get('confusion_matrix_path'),
                    'f1_scores': agg_results.get('f1_scores_path'),
                    'pattern_analysis': agg_results.get('pattern_analysis_path'),
                    'metrics': agg_results.get('metrics_path')
                }
        
        self.log_info("process_after_pipeline", "Confusion matrix analysis completed", {
            "total_files": summary['total_files_analyzed'],
            "successful": summary['successful_analyses'],
            "failed": summary['failed_analyses']
        })
        
        return {
            'summary': summary,
            'confusion_matrix_results': cm_results,
            'visualization_results': viz_results,
            'error_analysis_results': error_results,
            'checkpoint_dir': pipeline_output.get('checkpoint_dir')
        }

    @staticmethod
    def calculate_f1_scores(cm_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate F1 scores for each class using one-vs-all approach.
        
        Args:
            cm_df: Confusion matrix DataFrame
            
        Returns:
            Dictionary mapping class names to their F1 scores
        """
        f1_scores = {}
        total_samples = cm_df.sum().sum()
        
        for label in cm_df.index:
            # For one-vs-all:
            # True Positives: Correct predictions of the current class
            tp = cm_df.loc[label, label]
            
            # False Positives: All other predictions as this class
            fp = cm_df[label].sum() - tp
            
            # False Negatives: All predictions of this class as other classes
            fn = cm_df.loc[label].sum() - tp
            
            # True Negatives: All correct predictions of other classes
            tn = total_samples - (tp + fp + fn)
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores[label] = f1
            
        return f1_scores
        
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pipeline output data.
        Computes confusion matrices and saves results to checkpoint folder.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dictionary with confusion matrix results and file paths
        """
        self.log_info("process_after_pipeline", "Computing confusion matrices")
        
        loaded_files = pipeline_output.get('loaded_files', {})
        results = {}
        
        # Get checkpoint directory for saving results
        base_checkpoint_dir = self.path_manager.get_checkpoint_path(
            self.session_id,
            "confusion_matrix_analysis"
        )
        checkpoint_dir = base_checkpoint_dir / "results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize aggregated matrix
        all_labels = set()
        aggregated_cm = None
        
        for file_name, file_data in loaded_files.items():
            try:
                gt_df = file_data['gt_df']
                pred_df = file_data['pred_df']
                common_columns = file_data['common_columns']
                
                # Combine all columns into single series
                gt_combined = pd.concat([gt_df[col] for col in common_columns])
                pred_combined = pd.concat([pred_df[col] for col in common_columns])
                
                # Normalize values: convert to lowercase and strip whitespace
                gt_combined = gt_combined.astype(str).str.lower().str.strip()
                pred_combined = pred_combined.astype(str).str.lower().str.strip()
                
                # Replace empty strings and NaN with a placeholder
                gt_combined = gt_combined.replace(['', 'nan', 'none'], 'empty')
                pred_combined = pred_combined.replace(['', 'nan', 'none'], 'empty')
                
                # Get unique classes
                file_labels = sorted(set(gt_combined.unique()) | set(pred_combined.unique()))
                all_labels.update(file_labels)
                
                # Compute confusion matrix
                cm = confusion_matrix(
                    gt_combined,
                    pred_combined,
                    labels=file_labels
                )
                
                # Convert to DataFrame
                cm_df = pd.DataFrame(
                    cm,
                    index=file_labels,
                    columns=file_labels
                )
                
                # Calculate F1 scores
                f1_scores = self.calculate_f1_scores(cm_df)
                f1_scores_series = pd.Series(f1_scores)
                
                # Analyze confusion patterns
                pattern_analysis = self.pattern_analyzer.analyze_confusion_patterns(
                    cm_df,
                    f1_scores_series
                )
                
                # Save confusion matrix
                cm_path = checkpoint_dir / f"{file_name}_confusion_matrix.csv"
                cm_df.to_csv(cm_path)
                
                # Save F1 scores
                f1_path = checkpoint_dir / f"{file_name}_f1_scores.csv"
                f1_scores_series.to_csv(f1_path)
                
                # Save pattern analysis
                pattern_path = checkpoint_dir / f"{file_name}_pattern_analysis.json"
                pattern_dict = {
                    'problem_classes': {
                        k: {
                            'f1_score': v.f1_score,
                            'confused_with': {
                                k2: {
                                    'count': v2.count,
                                    'rate': v2.rate,
                                    'reverse_count': v2.reverse_count,
                                    'reverse_rate': v2.reverse_rate,
                                    'is_symmetric': v2.is_symmetric
                                } for k2, v2 in v.confused_with.items()
                            }
                        } for k, v in pattern_analysis.problem_classes.items()
                    },
                    'confusion_patterns': {
                        'most_common': [
                            {
                                'class_a': p.class_a,
                                'class_b': p.class_b,
                                'f1_a': p.f1_a,
                                'f1_b': p.f1_b,
                                'confusion_rate': p.confusion_rate,
                                'reverse_rate': p.reverse_rate,
                                'is_symmetric': p.is_symmetric
                            } for p in pattern_analysis.confusion_patterns['most_common']
                        ]
                    }
                }
                pd.Series(pattern_dict).to_json(pattern_path)
                
                # Compute classification metrics
                report = classification_report(
                    gt_combined,
                    pred_combined,
                    labels=file_labels,
                    output_dict=True
                )
                
                # Save metrics
                metrics_path = checkpoint_dir / f"{file_name}_metrics.csv"
                pd.DataFrame(report).transpose().to_csv(metrics_path)
                
                # Add to aggregated matrix
                if aggregated_cm is None:
                    aggregated_cm = cm_df
                else:
                    # Ensure same index and columns
                    cm_df = cm_df.reindex(index=all_labels, columns=all_labels, fill_value=0)
                    aggregated_cm = aggregated_cm.reindex(index=all_labels, columns=all_labels, fill_value=0)
                    aggregated_cm += cm_df
                
                results[file_name] = {
                    'status': 'success',
                    'confusion_matrix_path': str(cm_path),
                    'f1_scores_path': str(f1_path),
                    'pattern_analysis_path': str(pattern_path),
                    'metrics_path': str(metrics_path),
                    'classes': file_labels,
                    'columns_analyzed': common_columns
                }
                
                self.log_info("process_after_pipeline", 
                    f"Successfully computed confusion matrix for {file_name}",
                    {"columns_analyzed": len(common_columns)}
                )
                
            except Exception as e:
                self.log_error("process_after_pipeline", 
                    f"Error computing confusion matrix for {file_name}: {str(e)}")
                results[file_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Save aggregated matrix if we have results
        if aggregated_cm is not None:
            try:
                # Save aggregated confusion matrix
                agg_cm_path = checkpoint_dir / "aggregated_confusion_matrix.csv"
                aggregated_cm.to_csv(agg_cm_path)
                
                # Calculate and save aggregated F1 scores
                agg_f1_scores = self.calculate_f1_scores(aggregated_cm)
                agg_f1_scores_series = pd.Series(agg_f1_scores)
                agg_f1_path = checkpoint_dir / "aggregated_f1_scores.csv"
                agg_f1_scores_series.to_csv(agg_f1_path)
                
                # Analyze and save aggregated patterns
                agg_pattern_analysis = self.pattern_analyzer.analyze_confusion_patterns(
                    aggregated_cm,
                    agg_f1_scores_series
                )
                agg_pattern_path = checkpoint_dir / "aggregated_pattern_analysis.json"
                
                # Convert aggregated pattern analysis to dictionary format
                agg_pattern_dict = {
                    'problem_classes': {
                        k: {
                            'f1_score': v.f1_score,
                            'confused_with': {
                                k2: {
                                    'count': v2.count,
                                    'rate': v2.rate,
                                    'reverse_count': v2.reverse_count,
                                    'reverse_rate': v2.reverse_rate,
                                    'is_symmetric': v2.is_symmetric
                                } for k2, v2 in v.confused_with.items()
                            }
                        } for k, v in agg_pattern_analysis.problem_classes.items()
                    },
                    'confusion_patterns': {
                        'most_common': [
                            {
                                'class_a': p.class_a,
                                'class_b': p.class_b,
                                'f1_a': p.f1_a,
                                'f1_b': p.f1_b,
                                'confusion_rate': p.confusion_rate,
                                'reverse_rate': p.reverse_rate,
                                'is_symmetric': p.is_symmetric
                            } for p in agg_pattern_analysis.confusion_patterns['most_common']
                        ]
                    }
                }
                pd.Series(agg_pattern_dict).to_json(agg_pattern_path)
                
                # Compute and save aggregated metrics
                agg_metrics_path = checkpoint_dir / "aggregated_metrics.csv"
                agg_report = classification_report(
                    pd.Series(aggregated_cm.index.repeat(aggregated_cm.sum(axis=1))),
                    pd.Series(aggregated_cm.columns.repeat(aggregated_cm.sum(axis=0))),
                    labels=sorted(all_labels),
                    output_dict=True
                )
                pd.DataFrame(agg_report).transpose().to_csv(agg_metrics_path)
                
                results['aggregated'] = {
                    'status': 'success',
                    'confusion_matrix_path': str(agg_cm_path),
                    'f1_scores_path': str(agg_f1_path),
                    'pattern_analysis_path': str(agg_pattern_path),
                    'metrics_path': str(agg_metrics_path),
                    'classes': sorted(all_labels)
                }
                
                self.log_info("process_after_pipeline", 
                    "Successfully computed aggregated confusion matrix")
                
            except Exception as e:
                self.log_error("process_after_pipeline", 
                    f"Error computing aggregated confusion matrix: {str(e)}")
                results['aggregated'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return {
            **pipeline_output,
            'confusion_matrix_results': results,
            'checkpoint_dir': str(checkpoint_dir)
        } 
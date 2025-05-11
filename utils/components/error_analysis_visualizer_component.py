from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from utils.pipeline_component import PipelineComponent
from utils.data_classes import ConfusionAnalysis

@dataclass
class ErrorVisualizationConfig:
    """Configuration for error analysis visualizations."""
    figure_size: tuple = (15, 10)
    dpi: int = 300
    cmap: str = 'Reds'  # Using Reds for error visualization
    annot_fmt: str = 'd'
    title_fontsize: int = 14
    label_fontsize: int = 12
    annot_fontsize: int = 10
    bar_width: float = 0.35
    error_colors: List[str] = field(default_factory=lambda: ['#ff9999', '#66b3ff', '#99ff99'])  # Red, Blue, Green

class ErrorAnalysisVisualizerComponent(PipelineComponent):
    """
    Component for analyzing and visualizing error patterns in confusion matrices.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        enable_logging: bool = True,
        enable_console: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        operation_name: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the error analysis visualizer component.
        
        Args:
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
            **kwargs: Additional keyword arguments for component initialization
        """
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "error_analysis_visualizer",
            **kwargs
        )
        
        self.path_manager = None
        self.config = ErrorVisualizationConfig()  # Initialize the config

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data before running the pipeline.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dictionary with prepared data for the pipeline
        """
        # Get path_manager from input data
        self.path_manager = input_data.get('path_manager')
        
        if not self.path_manager:
            self.log_warning("process_before_pipeline", "No path manager found in input data")
            
        return {
            **input_data,
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
        
        self.log_info("process_after_pipeline", "Error analysis visualization completed", {
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

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load the confusion matrix results for error analysis."""
        self.log_info("process_before_pipeline", "Loading confusion matrix results for error analysis")
        
        loaded_matrices = {}
        for file_name, file_results in input_data.get('confusion_matrix_results', {}).items():
            if file_results['status'] == 'success':
                try:
                    # Load confusion matrix
                    cm_path = file_results['confusion_matrix_path']
                    cm_df = pd.read_csv(cm_path, index_col=0)
                    
                    # Load F1 scores
                    f1_path = file_results['f1_scores_path']
                    f1_scores = pd.read_csv(f1_path, index_col=0).squeeze()
                    
                    # Load pattern analysis
                    pattern_path = file_results['pattern_analysis_path']
                    pattern_dict = pd.read_json(pattern_path, typ='series').to_dict()
                    
                    loaded_matrices[file_name] = {
                        'confusion_matrix': cm_df,
                        'f1_scores': f1_scores,
                        'pattern_analysis': pattern_dict,
                        'classes': file_results.get('classes', [])
                    }
                    
                    self.log_info("process_before_pipeline", 
                        f"Successfully loaded results for {file_name}")
                    
                except Exception as e:
                    self.log_error("process_before_pipeline", 
                        f"Error loading results for {file_name}: {str(e)}")
        
        return {
            **input_data,
            'loaded_matrices': loaded_matrices
        }
    
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed error analysis visualizations."""
        self.log_info("process_after_pipeline", "Generating error analysis visualizations")
        
        loaded_matrices = pipeline_output.get('loaded_matrices', {})
        results = {}
        
        # Get checkpoint directory for saving results
        base_checkpoint_dir = self.path_manager.get_checkpoint_path(
            self.session_id,
            "confusion_matrix_analysis"
        )
        checkpoint_dir = base_checkpoint_dir / "error_analysis"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for file_name, matrix_data in loaded_matrices.items():
            try:
                cm_df = matrix_data['confusion_matrix']
                f1_scores = matrix_data['f1_scores']
                pattern_analysis = matrix_data['pattern_analysis']
                
                # 1. Create detailed error analysis for worst performers
                problem_classes = pattern_analysis['problem_classes']
                
                # Create a figure with subplots for each problem class
                n_classes = len(problem_classes)
                n_cols = min(3, n_classes)
                n_rows = (n_classes + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.config.figure_size[0], self.config.figure_size[1] * n_rows))
                axes = axes.flatten()
                
                for idx, (class_name, data) in enumerate(problem_classes.items()):
                    ax = axes[idx]
                    
                    # Get confusion data for this class
                    confused_with = data['confused_with']
                    other_classes = list(confused_with.keys())
                    rates = [confused_with[c]['rate'] for c in other_classes]
                    reverse_rates = [confused_with[c]['reverse_rate'] for c in other_classes]
                    
                    # Create grouped bar chart
                    x = np.arange(len(other_classes))
                    ax.bar(x - self.config.bar_width/2, rates, self.config.bar_width, 
                          label='Class → Other', color=self.config.error_colors[0])
                    ax.bar(x + self.config.bar_width/2, reverse_rates, self.config.bar_width,
                          label='Other → Class', color=self.config.error_colors[1])
                    
                    ax.set_title(f'Class: {class_name}\nF1 Score: {data["f1_score"]:.3f}',
                               fontsize=self.config.title_fontsize)
                    ax.set_xticks(x)
                    ax.set_xticklabels(other_classes, rotation=45, ha='right',
                                     fontsize=self.config.label_fontsize)
                    ax.set_ylabel('Confusion Rate', fontsize=self.config.label_fontsize)
                    ax.legend(fontsize=self.config.label_fontsize)
                    
                    # Add grid for better readability
                    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Remove empty subplots
                for idx in range(len(problem_classes), len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                
                # Save detailed error analysis
                error_analysis_path = checkpoint_dir / f"{file_name}_detailed_error_analysis.png"
                plt.savefig(error_analysis_path, bbox_inches='tight', dpi=self.config.dpi)
                plt.close()
                
                # 2. Create error type comparison
                plt.figure(figsize=self.config.figure_size)
                
                # Calculate error types for each problem class
                error_types = []
                for class_name, data in problem_classes.items():
                    total_errors = sum(conf['count'] for conf in data['confused_with'].values())
                    total_reverse_errors = sum(conf['reverse_count'] for conf in data['confused_with'].values())
                    error_types.append({
                        'class': class_name,
                        'misclassified_as_other': total_errors,
                        'other_misclassified_as': total_reverse_errors,
                        'f1_score': data['f1_score']
                    })
                
                error_df = pd.DataFrame(error_types)
                error_df = error_df.sort_values('f1_score')
                
                # Create stacked bar chart
                plt.bar(error_df['class'], error_df['misclassified_as_other'],
                       label='Class → Other', color=self.config.error_colors[0])
                plt.bar(error_df['class'], error_df['other_misclassified_as'],
                       bottom=error_df['misclassified_as_other'],
                       label='Other → Class', color=self.config.error_colors[1])
                
                plt.title('Error Type Distribution for Problem Classes',
                         fontsize=self.config.title_fontsize)
                plt.xlabel('Class', fontsize=self.config.label_fontsize)
                plt.ylabel('Number of Errors', fontsize=self.config.label_fontsize)
                plt.xticks(rotation=45, ha='right', fontsize=self.config.label_fontsize)
                plt.legend(fontsize=self.config.label_fontsize)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save error type comparison
                error_types_path = checkpoint_dir / f"{file_name}_error_types.png"
                plt.savefig(error_types_path, bbox_inches='tight', dpi=self.config.dpi)
                plt.close()
                
                # 3. Create confusion relationship heatmap
                plt.figure(figsize=(self.config.figure_size[0] * 1.5, self.config.figure_size[1] * 1.5))
                
                # Create a matrix of confusion rates between problem classes
                problem_class_list = list(problem_classes.keys())
                confusion_matrix = np.zeros((len(problem_class_list), len(problem_class_list)))
                
                # Set a minimum threshold for confusion rates to be considered
                min_confusion_threshold = 0.01  # 1% confusion rate
                
                for i, class_a in enumerate(problem_class_list):
                    for j, class_b in enumerate(problem_class_list):
                        if class_a != class_b:  # Skip self-confusion
                            # Get confusion data from class_a's perspective
                            if class_b in problem_classes[class_a]['confused_with']:
                                rate_a_to_b = problem_classes[class_a]['confused_with'][class_b]['rate']
                                if rate_a_to_b >= min_confusion_threshold:
                                    confusion_matrix[i, j] = rate_a_to_b
                            
                            # Also consider reverse confusion
                            if class_a in problem_classes[class_b]['confused_with']:
                                rate_b_to_a = problem_classes[class_b]['confused_with'][class_a]['rate']
                                if rate_b_to_a >= min_confusion_threshold:
                                    confusion_matrix[i, j] = max(
                                        confusion_matrix[i, j],
                                        rate_b_to_a
                                    )
                
                # Create heatmap with improved visualization
                plt.figure(figsize=(self.config.figure_size[0] * 1.5, self.config.figure_size[1] * 1.5))
                
                # Use a diverging colormap to better show the relationships
                sns.heatmap(
                    confusion_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',  # Red-Blue diverging colormap
                    xticklabels=problem_class_list,
                    yticklabels=problem_class_list,
                    annot_kws={'size': self.config.annot_fontsize},
                    vmin=min_confusion_threshold,  # Set minimum value to threshold
                    vmax=1.0,  # Set maximum value to 1
                    center=0.5,  # Center the colormap at 0.5
                    mask=confusion_matrix < min_confusion_threshold  # Mask values below threshold
                )
                
                plt.title('Confusion Relationships Between Problem Classes\n'
                         f'(Only showing confusion rates ≥ {min_confusion_threshold:.0%})',
                         fontsize=self.config.title_fontsize)
                plt.xlabel('Predicted Class', fontsize=self.config.label_fontsize)
                plt.ylabel('True Class', fontsize=self.config.label_fontsize)
                plt.xticks(rotation=45, ha='right', fontsize=self.config.label_fontsize)
                plt.yticks(rotation=0, fontsize=self.config.label_fontsize)
                
                # Add a colorbar label
                cbar = plt.gca().collections[0].colorbar
                cbar.set_label('Confusion Rate', fontsize=self.config.label_fontsize)
                
                plt.tight_layout()
                
                # Save confusion relationship heatmap
                confusion_heatmap_path = checkpoint_dir / f"{file_name}_confusion_relationships.png"
                plt.savefig(confusion_heatmap_path, bbox_inches='tight', dpi=self.config.dpi)
                plt.close()
                
                results[file_name] = {
                    'status': 'success',
                    'detailed_error_analysis_path': str(error_analysis_path),
                    'error_types_path': str(error_types_path),
                    'confusion_relationships_path': str(confusion_heatmap_path)
                }
                
                self.log_info("process_after_pipeline", 
                    f"Successfully generated error analysis visualizations for {file_name}")
                
            except Exception as e:
                self.log_error("process_after_pipeline", 
                    f"Error generating error analysis visualizations for {file_name}: {str(e)}")
                results[file_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return {
            **pipeline_output,
            'error_analysis_results': results,
            'checkpoint_dir': str(checkpoint_dir)
        } 
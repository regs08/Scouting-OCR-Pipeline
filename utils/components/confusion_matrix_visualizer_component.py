from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from utils.pipeline_component import PipelineComponent
from utils.data_classes import ConfusionAnalysis, ClassConfusion, ProblemClass, ConfusionPattern

@dataclass
class VisualizationConfig:
    """Configuration for confusion matrix visualizations."""
    figure_size: tuple = (10, 8)
    dpi: int = 300
    cmap: str = 'Blues'
    annot_fmt: str = 'd'
    show_colorbar: bool = True
    title_fontsize: int = 12
    label_fontsize: int = 10
    annot_fontsize: int = 8
    
    # Configuration for aggregated visualizations
    agg_figure_size: tuple = (20, 16)  # Larger figure size for aggregated plots
    agg_title_fontsize: int = 16
    agg_label_fontsize: int = 14
    agg_annot_fontsize: int = 12
    agg_tick_rotation: int = 45  # Rotate labels for better readability

class ConfusionMatrixVisualizerComponent(PipelineComponent):
    """
    Component that visualizes confusion matrices and generates summary reports.
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
        Initialize the confusion matrix visualizer component.
        
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
            operation_name=operation_name or "confusion_matrix_visualizer",
            **kwargs
        )
        
        self.path_manager = None
        self.config = VisualizationConfig()
        
    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data before running the pipeline.
        Loads the confusion matrix results and prepares them for visualization.
        
        Args:
            input_data: Input data dictionary containing confusion matrix results
            
        Returns:
            Dictionary with loaded confusion matrices and metadata
        """
        self.log_info("process_before_pipeline", "Loading confusion matrix results")
        
        # Get path_manager from input data
        self.path_manager = input_data.get('path_manager')
        
        if not self.path_manager:
            self.log_warning("process_before_pipeline", "No path manager found in input data")
        
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
                    
                    # Convert pattern analysis back to ConfusionAnalysis object
                    problem_classes = {}
                    for class_name, data in pattern_dict['problem_classes'].items():
                        confused_with = {}
                        for other_class, conf_data in data['confused_with'].items():
                            confused_with[other_class] = ClassConfusion(
                                count=conf_data['count'],
                                rate=conf_data['rate'],
                                reverse_count=conf_data['reverse_count'],
                                reverse_rate=conf_data['reverse_rate'],
                                is_symmetric=conf_data['is_symmetric']
                            )
                        problem_classes[class_name] = ProblemClass(
                            f1_score=data['f1_score'],
                            confused_with=confused_with
                        )
                    
                    confusion_patterns = {
                        'most_common': [
                            ConfusionPattern(
                                class_a=p['class_a'],
                                class_b=p['class_b'],
                                f1_a=p['f1_a'],
                                f1_b=p['f1_b'],
                                confusion_rate=p['confusion_rate'],
                                reverse_rate=p['reverse_rate'],
                                is_symmetric=p['is_symmetric']
                            ) for p in pattern_dict['confusion_patterns']['most_common']
                        ]
                    }
                    
                    pattern_analysis = ConfusionAnalysis(
                        problem_classes=problem_classes,
                        confusion_patterns=confusion_patterns
                    )
                    
                    # Load metrics
                    metrics_path = file_results['metrics_path']
                    metrics_df = pd.read_csv(metrics_path, index_col=0)
                    
                    loaded_matrices[file_name] = {
                        'confusion_matrix': cm_df,
                        'f1_scores': f1_scores,
                        'pattern_analysis': pattern_analysis,
                        'metrics': metrics_df,
                        'classes': file_results.get('classes', []),
                        'columns_analyzed': file_results.get('columns_analyzed', [])
                    }
                    
                    self.log_info("process_before_pipeline", 
                        f"Successfully loaded results for {file_name}")
                    
                except Exception as e:
                    self.log_error("process_before_pipeline", 
                        f"Error loading results for {file_name}: {str(e)}")
        
        return {
            **input_data,
            'loaded_matrices': loaded_matrices,
            'path_manager': self.path_manager
        }
        
    def visualize_accuracy_performance(
        self,
        cm_df: pd.DataFrame,
        title: str = "Class Accuracy Performance",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        top_n: int = 20
    ) -> Optional[str]:
        """
        Visualize accuracy performance for each class.
        
        Args:
            cm_df: Confusion matrix DataFrame
            title: Title for the plot
            figsize: Figure size for the visualization
            save_path: Path to save the visualization
            show: Whether to display the plot interactively
            top_n: Number of worst-performing classes to display
            
        Returns:
            Path to the saved visualization or None if not saved
        """
        # Calculate accuracy for each class
        accuracies = {}
        total_correct = 0
        total_predictions = 0
        
        for label in cm_df.index:
            # True positives (correct predictions)
            tp = cm_df.loc[label, label]
            # Total predictions for this class
            total = cm_df.loc[label].sum()
            # Calculate accuracy
            accuracy = tp / total if total > 0 else 0
            accuracies[label] = accuracy
            
            # Update totals for overall accuracy
            total_correct += tp
            total_predictions += total
        
        # Calculate overall accuracy
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        # Convert to DataFrame and sort by accuracy
        acc_df = pd.DataFrame({
            'Class': list(accuracies.keys()),
            'Accuracy': list(accuracies.values())
        })
        acc_df = acc_df.sort_values('Accuracy')  # Sort ascending to get worst performers first
        
        # Take top N worst performers
        acc_df = acc_df.head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(acc_df)), acc_df['Accuracy'])
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{acc_df["Accuracy"].iloc[i]:.1%}',
                va='center',
                fontsize=self.config.annot_fontsize
            )
        
        plt.yticks(range(len(acc_df)), acc_df['Class'], fontsize=self.config.label_fontsize)
        plt.xlabel('Accuracy', fontsize=self.config.label_fontsize)
        plt.title(f'{title}\nOverall Accuracy: {overall_accuracy:.1%}', fontsize=self.config.title_fontsize)
        
        # Add reference lines
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='50% Threshold')
        plt.axvline(x=overall_accuracy, color='g', linestyle='--', alpha=0.5, label='Overall Accuracy')
        plt.legend(fontsize=self.config.label_fontsize)
        
        # Set x-axis limits and format
        plt.xlim(0, 1.0)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
            plt.close()
            return str(save_path)
        elif show:
            plt.show()
            plt.close()
        
        return None

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pipeline output data.
        Generates visualizations and summary reports.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dictionary with visualization results and file paths
        """
        self.log_info("process_after_pipeline", "Generating visualizations and reports")
        
        loaded_matrices = pipeline_output.get('loaded_matrices', {})
        results = {}
        
        # Get checkpoint directory for saving results
        base_checkpoint_dir = self.path_manager.get_checkpoint_path(
            self.session_id,
            "confusion_matrix_analysis"
        )
        checkpoint_dir = base_checkpoint_dir / "visualizations"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations for each file
        for file_name, matrix_data in loaded_matrices.items():
            try:
                cm_df = matrix_data['confusion_matrix']
                f1_scores = matrix_data['f1_scores']
                pattern_analysis = matrix_data['pattern_analysis']
                metrics_df = matrix_data['metrics']
                
                # Create confusion matrix visualization
                is_aggregated = file_name == 'aggregated'
                plt.figure(figsize=self.config.agg_figure_size if is_aggregated else self.config.figure_size)
                
                # Create heatmap
                sns.heatmap(
                    cm_df,
                    annot=True,
                    fmt=self.config.annot_fmt,
                    cmap=self.config.cmap,
                    xticklabels=cm_df.columns,
                    yticklabels=cm_df.index,
                    annot_kws={'size': self.config.agg_annot_fontsize if is_aggregated else self.config.annot_fontsize}
                )
                
                # Set title and labels with appropriate font sizes
                plt.title(
                    f'Confusion Matrix - {file_name}',
                    fontsize=self.config.agg_title_fontsize if is_aggregated else self.config.title_fontsize
                )
                plt.xlabel(
                    'Predicted',
                    fontsize=self.config.agg_label_fontsize if is_aggregated else self.config.label_fontsize
                )
                plt.ylabel(
                    'True',
                    fontsize=self.config.agg_label_fontsize if is_aggregated else self.config.label_fontsize
                )
                
                # Rotate x-axis labels for better readability in aggregated plots
                if is_aggregated:
                    plt.xticks(rotation=self.config.agg_tick_rotation)
                    plt.yticks(rotation=0)
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                # Save confusion matrix visualization
                viz_path = checkpoint_dir / f"{file_name}_confusion_matrix.png"
                plt.savefig(viz_path, bbox_inches='tight', dpi=self.config.dpi)
                plt.close()
                
                # Create accuracy performance visualization
                accuracy_viz_path = checkpoint_dir / f"{file_name}_accuracy_performance.png"
                self.visualize_accuracy_performance(
                    cm_df,
                    title=f'Class Accuracy Performance - {file_name}',
                    figsize=self.config.agg_figure_size if is_aggregated else self.config.figure_size,
                    save_path=accuracy_viz_path,
                    show=False,
                    top_n=20
                )
                
                # Create F1 score visualization
                plt.figure(figsize=self.config.agg_figure_size if is_aggregated else self.config.figure_size)
                classes = f1_scores.index
                scores = f1_scores.values
                
                # Sort by F1 score
                sorted_indices = np.argsort(scores)
                classes = [classes[i] for i in sorted_indices]
                scores = [scores[i] for i in sorted_indices]
                
                # Create horizontal bar plot
                bars = plt.barh(range(len(classes)), scores)
                
                # Add value labels with appropriate font size
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{scores[i]:.3f}',
                        va='center',
                        fontsize=self.config.agg_annot_fontsize if is_aggregated else self.config.annot_fontsize
                    )
                
                plt.yticks(range(len(classes)), classes, fontsize=self.config.agg_label_fontsize if is_aggregated else self.config.label_fontsize)
                plt.xlabel(
                    'F1 Score',
                    fontsize=self.config.agg_label_fontsize if is_aggregated else self.config.label_fontsize
                )
                plt.title(
                    f'F1 Scores by Class - {file_name}',
                    fontsize=self.config.agg_title_fontsize if is_aggregated else self.config.title_fontsize
                )
                
                # Add a vertical line at F1 = 0.5
                plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save F1 score visualization
                f1_viz_path = checkpoint_dir / f"{file_name}_f1_scores.png"
                plt.savefig(f1_viz_path, bbox_inches='tight', dpi=self.config.dpi)
                plt.close()
                
                # Create confusion pattern visualization
                problem_classes = pattern_analysis.problem_classes
                most_common = pattern_analysis.confusion_patterns['most_common']
                
                if most_common:
                    # Create confusion network visualization
                    plt.figure(figsize=self.config.agg_figure_size if is_aggregated else self.config.figure_size)
                    
                    # Extract nodes and edges
                    nodes = set()
                    edges = []
                    for pattern in most_common:
                        nodes.add(pattern.class_a)
                        nodes.add(pattern.class_b)
                        edges.append((
                            pattern.class_a,
                            pattern.class_b,
                            pattern.confusion_rate + pattern.reverse_rate
                        ))
                    
                    # Create network plot
                    G = nx.Graph()
                    G.add_nodes_from(nodes)
                    G.add_weighted_edges_from(edges)
                    
                    # Calculate node positions with more space
                    pos = nx.spring_layout(G, k=2, iterations=50)
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(
                        G, pos,
                        node_color='lightblue',
                        node_size=2000 if is_aggregated else 1000,
                        alpha=0.6
                    )
                    
                    # Draw edges with weights
                    nx.draw_networkx_edges(
                        G, pos,
                        width=[G[u][v]['weight'] * 5 for u, v in G.edges()],
                        alpha=0.4
                    )
                    
                    # Add labels with appropriate font size
                    nx.draw_networkx_labels(
                        G, pos,
                        font_size=self.config.agg_label_fontsize if is_aggregated else self.config.label_fontsize
                    )
                    
                    plt.title(
                        f'Confusion Network - {file_name}',
                        fontsize=self.config.agg_title_fontsize if is_aggregated else self.config.title_fontsize
                    )
                    plt.axis('off')
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save confusion network visualization
                    network_path = checkpoint_dir / f"{file_name}_confusion_network.png"
                    plt.savefig(network_path, bbox_inches='tight', dpi=self.config.dpi)
                    plt.close()
                
                # Save metrics report
                report_path = checkpoint_dir / f"{file_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(f"Confusion Matrix Analysis Report - {file_name}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Columns Analyzed: {matrix_data['columns_analyzed']}\n\n")
                    
                    f.write("Classification Metrics:\n")
                    f.write(metrics_df.to_string())
                    f.write("\n\nF1 Scores by Class:\n")
                    for class_name, score in f1_scores.items():
                        f.write(f"{class_name}: {score:.3f}\n")
                    
                    f.write("\n\nConfusion Pattern Analysis:\n")
                    f.write("-" * 30 + "\n")
                    
                    for class_name, data in problem_classes.items():
                        f.write(f"\nProblem Class: {class_name}\n")
                        f.write(f"F1 Score: {data.f1_score:.3f}\n")
                        f.write("Confused with:\n")
                        for other_class, conf_data in data.confused_with.items():
                            f.write(f"  - {other_class}:\n")
                            f.write(f"    * Rate: {conf_data.rate:.3f}\n")
                            f.write(f"    * Reverse Rate: {conf_data.reverse_rate:.3f}\n")
                            f.write(f"    * Symmetric: {conf_data.is_symmetric}\n")
                
                results[file_name] = {
                    'status': 'success',
                    'confusion_matrix_path': str(viz_path),
                    'f1_scores_path': str(f1_viz_path),
                    'confusion_network_path': str(network_path) if most_common else None,
                    'report_path': str(report_path)
                }
                
                self.log_info("process_after_pipeline", 
                    f"Successfully generated visualizations for {file_name}")
                
            except Exception as e:
                self.log_error("process_after_pipeline", 
                    f"Error generating visualizations for {file_name}: {str(e)}")
                results[file_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Generate summary report
        try:
            summary_path = checkpoint_dir / "summary_report.txt"
            with open(summary_path, 'w') as f:
                f.write("Confusion Matrix Analysis Summary\n")
                f.write("=" * 30 + "\n\n")
                
                for file_name, result in results.items():
                    if file_name == 'summary':
                        continue
                        
                    f.write(f"\nFile: {file_name}\n")
                    f.write("-" * 20 + "\n")
                    
                    if result['status'] == 'success':
                        f.write(f"Confusion Matrix: {result['confusion_matrix_path']}\n")
                        f.write(f"F1 Scores: {result['f1_scores_path']}\n")
                        if result.get('confusion_network_path'):
                            f.write(f"Confusion Network: {result['confusion_network_path']}\n")
                        f.write(f"Report: {result['report_path']}\n")
                    else:
                        f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            
            results['summary'] = {
                'status': 'success',
                'summary_path': str(summary_path)
            }
            
            self.log_info("process_after_pipeline", "Successfully generated summary report")
            
        except Exception as e:
            self.log_error("process_after_pipeline", 
                f"Error generating summary report: {str(e)}")
            results['summary'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return {
            **pipeline_output,
            'visualization_results': results,
            'checkpoint_dir': str(checkpoint_dir)
        } 
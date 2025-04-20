import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

class ConfusionMatrixVisualizer:
    """
    Class for creating visualizations of confusion matrices.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the confusion matrix visualizer.
        
        Args:
            output_dir: Directory where visualizations will be saved
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "confusion_matrix_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize(
        self,
        confusion_matrix: pd.DataFrame,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        max_display_size: int = 30,
        cmap: str = 'Blues'
    ) -> Optional[str]:
        """
        Visualize a confusion matrix.
        
        Args:
            confusion_matrix: DataFrame containing the confusion matrix
            title: Title for the plot
            figsize: Base figure size for the visualization
            save_path: Path to save the visualization (if None, will use output_dir)
            show: Whether to display the plot interactively
            max_display_size: Maximum number of rows/columns to display
            cmap: Colormap to use for the heatmap
            
        Returns:
            Path to the saved visualization or None if not saved
        """
        # Limit the size of the matrix for better visualization
        if confusion_matrix.shape[0] > max_display_size or confusion_matrix.shape[1] > max_display_size:
            # Get row and column sums
            row_sums = confusion_matrix.sum(axis=1)
            col_sums = confusion_matrix.sum(axis=0)
            
            # Keep top values by frequency
            top_rows = row_sums.nlargest(max_display_size).index
            top_cols = col_sums.nlargest(max_display_size).index
            
            # Filter the matrix
            filtered_matrix = confusion_matrix.loc[top_rows, top_cols]
            title += f" (Top {max_display_size} values)"
        else:
            filtered_matrix = confusion_matrix
        
        # Calculate correct predictions for accuracy in title
        total_predictions = filtered_matrix.sum().sum()
        correct_predictions = sum(filtered_matrix.loc[val, val] 
                                for val in filtered_matrix.index 
                                if val in filtered_matrix.columns)
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        title += f" (Accuracy: {accuracy:.2f}%)"
        
        # Adjust figure size based on matrix dimensions
        matrix_size = max(filtered_matrix.shape)
        scale_factor = max(1, matrix_size / 10)  # Scale based on matrix size
        adjusted_figsize = (figsize[0] * scale_factor, figsize[1] * scale_factor)
        
        # Create the plot
        plt.figure(figsize=adjusted_figsize)
        
        # Use smaller font size for larger matrices
        font_size = max(6, 12 - (matrix_size // 10))
        
        # Create the heatmap
        ax = sns.heatmap(
            filtered_matrix,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            linewidths=0.5,
            annot_kws={"size": font_size},
            cbar_kws={"shrink": 0.7}
        )
        
        # Set labels and title
        plt.title(title, fontsize=12 + (4 / scale_factor))
        plt.ylabel('True Values', fontsize=10 + (2 / scale_factor))
        plt.xlabel('Predicted Values', fontsize=10 + (2 / scale_factor))
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
        
        # Tight layout to maximize figure area
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            full_path = Path(save_path)
        else:
            # Generate a filename based on the title
            filename = title.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
            filename = f"{filename}.png"
            full_path = self.output_dir / filename
            
        if save_path is not None or not show:
            plt.savefig(full_path, bbox_inches='tight', dpi=150)
            saved_path = str(full_path)
        else:
            saved_path = None
            
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return saved_path
        
    def visualize_class_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Class Performance Metrics",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        top_n: int = 20
    ) -> Optional[str]:
        """
        Visualize performance metrics for each class.
        
        Args:
            metrics: Dictionary mapping class names to their metrics
            title: Title for the plot
            figsize: Figure size for the visualization
            save_path: Path to save the visualization (if None, will use output_dir)
            show: Whether to display the plot interactively
            top_n: Number of top classes to display (by F1 score)
            
        Returns:
            Path to the saved visualization or None if not saved
        """
        # Convert metrics to DataFrame for easier plotting
        data = []
        for cls, cls_metrics in metrics.items():
            data.append({
                'Class': cls,
                'Precision': cls_metrics['precision'],
                'Recall': cls_metrics['recall'],
                'F1 Score': cls_metrics['f1_score']
            })
            
        df = pd.DataFrame(data)
        
        # Sort by F1 score and take top N
        df = df.sort_values('F1 Score', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Use a grouped bar chart
        x = np.arange(len(df))
        width = 0.25
        
        plt.bar(x - width, df['Precision'], width, label='Precision')
        plt.bar(x, df['Recall'], width, label='Recall')
        plt.bar(x + width, df['F1 Score'], width, label='F1 Score')
        
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(x, df['Class'], rotation=45, ha='right')
        plt.legend()
        
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            full_path = Path(save_path)
        else:
            # Generate a filename based on the title
            filename = title.replace(" ", "_") + ".png"
            full_path = self.output_dir / filename
            
        if save_path is not None or not show:
            plt.savefig(full_path, bbox_inches='tight', dpi=150)
            saved_path = str(full_path)
        else:
            saved_path = None
            
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return saved_path
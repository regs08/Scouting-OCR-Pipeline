from typing import List, Dict, Union, Any
from pathlib import Path
import pandas as pd
from .confusion_matrix_storage import ConfusionMatrixStorage

class ConfusionMatrixAggregator:
    """Simple class to aggregate multiple confusion matrices into one."""
    
    def __init__(self, storage: ConfusionMatrixStorage):
        """
        Initialize the aggregator.
        
        Args:
            storage: ConfusionMatrixStorage instance for loading/saving matrices
        """
        self.storage = storage
    
    def aggregate_matrices(
        self,
        matrix_files: List[Union[str, Path]],
        output_identifier: str = "aggregated"
    ) -> Dict[str, Any]:
        """
        Combine multiple confusion matrices into one, combining only exactly matching labels.
        
        Args:
            matrix_files: List of paths to confusion matrix files
            output_identifier: Identifier for the aggregated matrix
            
        Returns:
            Dictionary containing the aggregated matrix and basic stats
        """
        if not matrix_files:
            return {"status": "error", "message": "No matrix files provided"}
            
        try:
            # Load and sum all matrices
            aggregated_matrix = None
            processed_files = 0
            all_labels = set()
            
            # First pass: collect all unique labels
            for file_path in matrix_files:
                current_matrix = self.storage.load_matrix(file_path)
                # Convert all labels to strings for consistent comparison
                current_matrix.index = current_matrix.index.astype(str)
                current_matrix.columns = current_matrix.columns.astype(str)
                all_labels.update(current_matrix.index)
                all_labels.update(current_matrix.columns)
            
            # Sort labels
            sorted_labels = sorted(all_labels)
            
            # Initialize aggregated matrix with zeros
            aggregated_matrix = pd.DataFrame(
                0,
                index=sorted_labels,
                columns=sorted_labels,
                dtype=float
            )
            
            # Second pass: combine matrices
            for file_path in matrix_files:
                current_matrix = self.storage.load_matrix(file_path)
                current_matrix.index = current_matrix.index.astype(str)
                current_matrix.columns = current_matrix.columns.astype(str)
                
                # Add values to aggregated matrix
                for i in current_matrix.index:
                    for j in current_matrix.columns:
                        if pd.notna(current_matrix.loc[i, j]):
                            aggregated_matrix.loc[i, j] += current_matrix.loc[i, j]
                
                processed_files += 1
            
            # Save the aggregated matrix
            save_path = self.storage.save_matrix(
                confusion_matrix=aggregated_matrix,
                identifier=f"{output_identifier}_combined"
            )
            
            return {
                "status": "success",
                "aggregated_matrix": aggregated_matrix,
                "total_predictions": int(aggregated_matrix.sum().sum()),
                "matrices_combined": processed_files,
                "save_path": save_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during aggregation: {str(e)}"
            } 
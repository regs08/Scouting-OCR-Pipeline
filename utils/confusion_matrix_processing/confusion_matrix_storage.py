import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List, Union, Any

class ConfusionMatrixStorage:
    """
    Class for storing confusion matrices and their associated metrics.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the confusion matrix storage.
        
        Args:
            output_dir: Directory where matrices and metrics will be saved
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "confusion_matrix_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_matrix(
        self,
        confusion_matrix: pd.DataFrame,
        identifier: str,
        format: str = 'csv'
    ) -> str:
        """
        Save a confusion matrix to file.
        
        Args:
            confusion_matrix: DataFrame containing the confusion matrix
            identifier: Unique identifier for the confusion matrix
            format: File format ('csv' or 'json')
            
        Returns:
            Path to the saved file
        """
        # Create filename
        if format.lower() == 'csv':
            filename = f"{identifier}_confusion_matrix.csv"
            file_path = self.output_dir / filename
            confusion_matrix.to_csv(file_path)
        elif format.lower() == 'json':
            filename = f"{identifier}_confusion_matrix.json"
            file_path = self.output_dir / filename
            # Convert to serializable format
            matrix_dict = {
                'index': list(confusion_matrix.index),
                'columns': list(confusion_matrix.columns),
                'data': confusion_matrix.values.tolist()
            }
            with open(file_path, 'w') as f:
                json.dump(matrix_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")
            
        return str(file_path)
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        identifier: str,
        format: str = 'csv'
    ) -> Dict[str, str]:
        """
        Save metrics to files.
        
        Args:
            metrics: Dictionary containing metrics
            identifier: Unique identifier for the metrics
            format: File format ('csv', 'json')
            
        Returns:
            Dictionary mapping metric types to their file paths
        """
        saved_files = {}
        
        # Handle overall metrics (non-nested dictionaries)
        overall_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        
        if overall_metrics:
            if format.lower() == 'csv':
                filename = f"{identifier}_overall_metrics.csv"
                file_path = self.output_dir / filename
                pd.DataFrame([overall_metrics]).to_csv(file_path, index=False)
            elif format.lower() == 'json':
                filename = f"{identifier}_overall_metrics.json"
                file_path = self.output_dir / filename
                with open(file_path, 'w') as f:
                    json.dump(overall_metrics, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")
                
            saved_files['overall'] = str(file_path)
        
        # Handle class metrics if present
        if 'class_metrics' in metrics and isinstance(metrics['class_metrics'], dict):
            class_metrics = metrics['class_metrics']
            
            # Convert class metrics to DataFrame
            class_data = []
            for cls, cls_metrics in class_metrics.items():
                row = {'class': cls}
                row.update(cls_metrics)
                class_data.append(row)
                
            class_df = pd.DataFrame(class_data)
            
            if format.lower() == 'csv':
                filename = f"{identifier}_class_metrics.csv"
                file_path = self.output_dir / filename
                class_df.to_csv(file_path, index=False)
            elif format.lower() == 'json':
                filename = f"{identifier}_class_metrics.json"
                file_path = self.output_dir / filename
                with open(file_path, 'w') as f:
                    json.dump(class_data, f, indent=2)
                    
            saved_files['class_metrics'] = str(file_path)
            
        return saved_files
    
    def load_matrix(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a confusion matrix from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame containing the confusion matrix
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            # For CSV, the first column is the index
            return pd.read_csv(file_path, index_col=0)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                matrix_dict = json.load(f)
                
            # Reconstruct DataFrame
            if isinstance(matrix_dict, dict) and 'index' in matrix_dict and 'columns' in matrix_dict:
                # Format is {'index': [...], 'columns': [...], 'data': [[...]]}
                return pd.DataFrame(
                    data=matrix_dict['data'],
                    index=matrix_dict['index'],
                    columns=matrix_dict['columns']
                )
            else:
                # Try to load as regular DataFrame JSON format
                return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def load_metrics(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load metrics from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing the metrics
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            # For CSV, metrics are stored as a single row
            df = pd.read_csv(file_path)
            if len(df) > 0:
                return df.iloc[0].to_dict()
            return {}
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

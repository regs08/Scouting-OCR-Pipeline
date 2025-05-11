from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

@dataclass
class MatchedFile:
    """Data class representing a matched pair of ground truth and prediction files."""
    gt_path: Union[str, Path]
    pred_path: Union[str, Path]
    normalized_name: str

@dataclass
class ConfusionPattern:
    """Data class representing a confusion pattern between two classes."""
    class_a: str
    class_b: str
    f1_a: float
    f1_b: float
    confusion_rate: float
    reverse_rate: float
    is_symmetric: bool

@dataclass
class ClassConfusion:
    """Data class representing confusion data for a specific class pair."""
    count: int
    rate: float
    reverse_count: int
    reverse_rate: float
    is_symmetric: bool

@dataclass
class ProblemClass:
    """Data class representing a problem class with its F1 score and confusion data."""
    f1_score: float
    confused_with: Dict[str, ClassConfusion]

@dataclass
class ConfusionAnalysis:
    """Data class representing the complete confusion analysis results."""
    problem_classes: Dict[str, ProblemClass]
    confusion_patterns: Dict[str, List[ConfusionPattern]]

@dataclass
class ConfusionMatrixAnalysisResults:
    """
    Data class that encapsulates the results from the confusion matrix analysis pipeline.
    """
    summary: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    visualization_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_analysis_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    checkpoint_dir: Optional[str] = None
    
    @property
    def total_files_analyzed(self) -> int:
        """Get the total number of files analyzed."""
        return self.summary.get('total_files_analyzed', 0)
    
    @property
    def successful_analyses(self) -> int:
        """Get the number of successful analyses."""
        return self.summary.get('successful_analyses', 0)
    
    @property
    def failed_analyses(self) -> int:
        """Get the number of failed analyses."""
        return self.summary.get('failed_analyses', 0)
    
    @property
    def has_aggregated_results(self) -> bool:
        """Check if aggregated results are available."""
        return 'aggregated_results' in self.summary
    
    def get_aggregated_results(self) -> Optional[Dict[str, str]]:
        """Get the paths to aggregated results if available."""
        return self.summary.get('aggregated_results')
    
    def get_visualization_paths(self, file_name: str) -> Optional[Dict[str, str]]:
        """Get the paths to visualizations for a specific file."""
        if file_name in self.visualization_results:
            result = self.visualization_results[file_name]
            if result['status'] == 'success':
                return {
                    'confusion_matrix': result['confusion_matrix_path'],
                    'f1_scores': result['f1_scores_path'],
                    'confusion_network': result.get('confusion_network_path'),
                    'report': result['report_path']
                }
        return None
    
    def get_error_analysis_paths(self, file_name: str) -> Optional[Dict[str, str]]:
        """Get the paths to error analysis results for a specific file."""
        if file_name in self.error_analysis_results:
            result = self.error_analysis_results[file_name]
            if result['status'] == 'success':
                return {
                    'detailed_error_analysis': result['detailed_error_analysis_path'],
                    'error_types': result['error_types_path'],
                    'confusion_relationships': result['confusion_relationships_path']
                }
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary."""
        return {
            'summary': self.summary,
            'confusion_matrix_results': self.confusion_matrix_results,
            'visualization_results': self.visualization_results,
            'error_analysis_results': self.error_analysis_results,
            'checkpoint_dir': self.checkpoint_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfusionMatrixAnalysisResults':
        """Create a ConfusionMatrixAnalysisResults instance from a dictionary."""
        return cls(
            summary=data.get('summary', {}),
            confusion_matrix_results=data.get('confusion_matrix_results', {}),
            visualization_results=data.get('visualization_results', {}),
            error_analysis_results=data.get('error_analysis_results', {}),
            checkpoint_dir=data.get('checkpoint_dir')
        ) 
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

from utils.pipeline_component import PipelineComponent
from utils.data_classes import ConfusionMatrixAnalysisResults
from utils.path_manager import PathManager
from utils.runnable_component_config import RunnableComponentConfig

class ConfusionMatrixManager(PipelineComponent):
    """
    High-level component that coordinates the confusion matrix analysis pipeline.
    Manages the flow of data between components and provides a unified interface.
    """
    
    def __init__(
        self,
        *,
        component_configs: Optional[List[RunnableComponentConfig]] = None,
        path_manager: Optional[PathManager] = None,
        verbose: bool = True,
        enable_logging: bool = True,
        enable_console: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        operation_name: Optional[str] = None,
        top_n_problem_classes: int = 20,
        confusion_threshold: float = 0.01,
        **kwargs: Any
    ):
        """
        Initialize the confusion matrix manager.
        
        Args:
            component_configs: List of components to initialize
            path_manager: PathManager instance for managing file paths
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
            top_n_problem_classes: Number of worst-performing classes to analyze
            confusion_threshold: Minimum confusion rate to consider (0-1)
            **kwargs: Additional keyword arguments for component initialization
        """
        # Create session ID if not provided
        session_id = kwargs.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Initialize pipeline component with logging
        super().__init__(
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            component_configs=component_configs,
            operation_name=operation_name or "confusion_matrix_manager",
            **kwargs
        )
        
        self.path_manager = path_manager
        self.top_n_problem_classes = top_n_problem_classes
        self.confusion_threshold = confusion_threshold
        self.matched_files = []
        
        # Log initialization
        self.log_info("__init__", "Confusion matrix manager initialized", {
            "top_n_problem_classes": self.top_n_problem_classes,
            "confusion_threshold": self.confusion_threshold,
            "session_id": self.session_id
        })

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the confusion matrix analysis context for all pipeline components.
        """
        # Get required data from input
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        
        if not self.path_manager:
            raise ValueError("path_manager is required in input_data")
        if not self.session_id:
            raise ValueError("session_id is required in input_data")
            
        # Get session paths
        session_paths = self.path_manager.get_session_paths(self.session_id)
        
        # Create checkpoint directory
        checkpoint_dir = self.path_manager.get_checkpoint_path(
            self.session_id,
            "confusion_matrix_analysis"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get and verify matched files
        self.matched_files = input_data.get('matched_files', [])
        
        # Log detailed matched files information
        if self.matched_files:
            self.log_info("process_before_pipeline", "Received matched files details", {
                'num_matched_files': len(self.matched_files),
                'type': str(type(self.matched_files)),
                'first_file_type': str(type(self.matched_files[0])),
                'first_file_attrs': dir(self.matched_files[0]),
                'first_file_gt_path': str(self.matched_files[0].gt_path),
                'first_file_pred_path': str(self.matched_files[0].pred_path),
                'first_file_normalized_name': self.matched_files[0].normalized_name
            })
        else:
            self.log_warning("process_before_pipeline", "No matched files received", {
                'input_data_keys': list(input_data.keys())
            })
        
        # Prepare pipeline data with all necessary context
        pipeline_data = {
            **input_data,
            'path_manager': self.path_manager,
            'session_id': self.session_id,
            'session_paths': session_paths,
            'checkpoint_dir': checkpoint_dir,
            'top_n_problem_classes': self.top_n_problem_classes,
            'confusion_threshold': self.confusion_threshold,
            'matched_files': self.matched_files  # Ensure matched files are passed through
        }
        
        # Initialize sub-components with path manager
        for component_info in self.pipeline:
            component = component_info['component']
            if hasattr(component, 'path_manager'):
                component.path_manager = self.path_manager
            if hasattr(component, 'session_id'):
                component.session_id = self.session_id
        
        self.log_info("process_before_pipeline", "Confusion matrix pipeline preparation complete", {
            'session_id': self.session_id,
            'directories': list(session_paths.keys()),
            'checkpoint_dir': str(checkpoint_dir)
        })
        
        return pipeline_data
    
    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pipeline output data and prepare final results.
        
        Args:
            pipeline_output: Output data from the pipeline
            
        Returns:
            Dictionary containing the analysis results
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
        
        # Return dictionary instead of ConfusionMatrixAnalysisResults object
        return {
            **pipeline_output,
            'summary': summary,
            'confusion_matrix_results': cm_results,
            'visualization_results': viz_results,
            'error_analysis_results': error_results,
            'checkpoint_dir': pipeline_output.get('checkpoint_dir'),
            'analysis_completed': True
        } 
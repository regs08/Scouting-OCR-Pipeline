from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
import os
import numpy as np

from .base_comparison_processor import BaseComparisonProcessor
from utils.confusion_matrix_processing.confusion_matrix_analyzer import ConfusionMatrixAnalyzer

class ConfusionMatrixSessionProcessor(BaseComparisonProcessor):
    """
    Processor for comparing previous checkpoint data against ground truth files
    and generating confusion matrices for analysis.
    """
    def __init__(self,
                 path_manager,
                 session_id: str,
                 source_checkpoint_name: str = "ckpt3_column_correction",  # Source checkpoint
                 cols_to_process: Optional[List[str]] = None,  # New parameter
                 case_sensitive: bool = False,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
                 operation_name: Optional[str] = None):
        """
        Initialize the confusion matrix processor.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            source_checkpoint_name: Name of the checkpoint to compare against ground truth
            cols_to_process: List of specific columns to analyze (e.g., ['L1', 'L2', 'L3'])
            case_sensitive: Whether to perform case-sensitive comparisons
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            source_checkpoint_name=source_checkpoint_name,
            case_sensitive=case_sensitive,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "confusion_matrix_analysis"
        )
        
        # Store checkpoint names and columns to process
        self.source_checkpoint_name = source_checkpoint_name
        self.output_checkpoint_name = "ckpt4_confusion_matrix"
        self.cols_to_process = cols_to_process or ['L1', 'L2', 'L3', 'L4', 'L5', 
                                                 'L6', 'L7', 'L8', 'L9', 'L10',
                                                 'L11', 'L12', 'L13', 'L14', 'L15',
                                                 'L16', 'L17', 'L18', 'L19', 'L20']
        
        # Create OUTPUT checkpoint directory for confusion matrix results
        self.output_dir = self.path_manager.get_checkpoint_path(
            session_id=self.session_id,
            checkpoint=self.output_checkpoint_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the ConfusionMatrixAnalyzer with the output directory
        self.analyzer = ConfusionMatrixAnalyzer(
            output_dir=str(self.output_dir),
            case_sensitive=case_sensitive
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare checkpoint data with ground truth and generate confusion matrices.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Dictionary with confusion matrix analysis results
        """
        try:
            # Validate and get directories
            session_dir = self._validate_input(input_data)
            source_checkpoint_dir = self.path_manager.get_checkpoint_path(
                session_id=self.session_id,
                checkpoint=self.source_checkpoint_name  # Use source_checkpoint_name
            )
            
            # Log paths for debugging
            self._log_paths(session_dir, source_checkpoint_dir)
            
            # Load and validate data
            gt_data = self._load_ground_truth(session_dir)
            checkpoint_data = self._load_checkpoint_data(source_checkpoint_dir)
            
            if not self._validate_data(gt_data, checkpoint_data):
                return self._create_error_response("Invalid or missing data")
            
            # Find matching files
            matching_files = self._find_matching_files(gt_data, checkpoint_data)
            
            if not matching_files:
                return self._create_error_response("No matching files found")
            
            # Process each matched file pair
            file_results = self._process_file_pairs(matching_files)
            
            # Create and save summary
            summary_results = self._create_summary(file_results)
            
            # Explicitly call finalize_analysis to aggregate and save final results
            aggregated_results = self.analyzer.finalize_analysis(
                identifier=f"{self.session_id}_overview"
            )
            
            if aggregated_results["status"] == "success":
                self.log_info("process", f"Successfully created aggregated matrix:")
                self.log_info("process", f"Matrix saved to: {aggregated_results['file_paths']['matrix_csv']}")
                self.log_info("process", f"Visualization saved to: {aggregated_results['file_paths']['visualizations']['matrix']}")
            else:
                self.log_error("process", f"Failed to create aggregated matrix: {aggregated_results.get('message', 'Unknown error')}")
            
            return {
                "confusion_matrix_status": "completed",
                "total_files": len(matching_files),
                "file_results": file_results,
                "summary": summary_results,
                "aggregated_results": aggregated_results,
                "confusion_matrix_dir": str(self.output_dir),
                "source_checkpoint": self.source_checkpoint_name,
                "output_checkpoint": self.output_checkpoint_name
            }
            
        except Exception as e:
            self.log_error("process", f"Error in confusion matrix processing: {str(e)}")
            return self._create_error_response(str(e))

    def _validate_input(self, input_data: Dict[str, Any]) -> Path:
        """Validate input data and return session directory."""
        if not input_data.get('session_dir'):
            raise ValueError("Missing session_dir in input data")
        return Path(input_data['session_dir'])

    def _log_paths(self, session_dir: Path, source_checkpoint_dir: Path) -> None:
        """Log important directory paths."""
        self.log_info("process", f"Session directory: {session_dir.absolute()}")
        self.log_info("process", f"Source checkpoint directory (reading from): {source_checkpoint_dir.absolute()}")
        self.log_info("process", f"Output checkpoint directory (writing to): {self.output_dir.absolute()}")

    def _validate_data(self, gt_data: Dict, checkpoint_data: Dict) -> bool:
        """Validate loaded data."""
        if not gt_data:
            self.log_error("process", "No ground truth data found")
            return False
        if not checkpoint_data:
            self.log_error("process", "No checkpoint data found")
            return False
        return True

    def _process_file_pairs(self, matching_files: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Process each pair of matching files."""
        file_results = {}
        
        for filename, data_pair in matching_files.items():
            try:
                gt_df = data_pair['gt']
                pred_df = data_pair['pred']
                
                # Filter for specified columns that exist in both DataFrames
                columns_to_analyze = [col for col in self.cols_to_process 
                                    if col in gt_df.columns and col in pred_df.columns]
                
                if not columns_to_analyze:
                    self.log_warning("process", 
                        f"No specified columns found in both DataFrames for {filename}")
                    file_results[filename] = {
                        "status": "error",
                        "error": "No matching columns to analyze"
                    }
                    continue
                
                # Use analyzer to process the file pair with specified columns
                analysis_results = self.analyzer.analyze_dataframes(
                    gt_df=gt_df,
                    pred_df=pred_df,
                    columns=columns_to_analyze,
                    identifier=filename,
                    save_results=True,
                    create_visualizations=True,
                    aggregate_results=True  # Make sure aggregation is enabled
                )
                
                if analysis_results.get('file_paths', {}).get('confusion_matrix'):
                    self.log_info("process", 
                        f"Matrix saved for {filename}: {analysis_results['file_paths']['confusion_matrix']}")
                
                # Store results with more detailed metrics
                file_results[filename] = {
                    "status": "completed",
                    "metrics": analysis_results['metrics'],
                    "file_paths": analysis_results['file_paths'],
                    "columns_analyzed": columns_to_analyze,
                    "total_columns": len(columns_to_analyze),
                    "total_rows": len(gt_df)
                }
                
                self.log_info("process", 
                    f"Successfully analyzed {filename} with {len(columns_to_analyze)} columns")
                
            except Exception as e:
                self.log_error("process", f"Error processing {filename}: {str(e)}")
                file_results[filename] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return file_results

    def _create_summary(self, file_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create and save analysis summary."""
        completed_files = sum(1 for r in file_results.values() if r['status'] == 'completed')
        error_files = sum(1 for r in file_results.values() if r['status'] == 'error')
        
        summary = {
            'total_files': len(file_results),
            'completed_files': completed_files,
            'error_files': error_files
        }
        
        # Save summary using storage component
        try:
            self.analyzer.storage.save_metrics(
                metrics=summary,
                identifier='confusion_matrix_summary',
                format='csv'
            )
        except Exception as e:
            self.log_error("process", f"Could not save summary: {str(e)}")
        
        return summary

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "confusion_matrix_status": "error",
            "error": error_message,
            "source_checkpoint": self.source_checkpoint_name,
            "output_checkpoint": self.output_checkpoint_name
        } 
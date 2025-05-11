import pandas as pd
import numpy as np
from pathlib import Path
from utils.managers.confusion_matrix_manager import ConfusionMatrixManager
from utils.data_classes import MatchedFile, ConfusionMatrixAnalysisResults
from utils.path_manager import PathManager

def convert_to_string(df):
    """Convert all values in numeric columns to strings without cleaning."""
    numeric_cols = [col for col in df.columns if col.startswith('L')]
    for col in numeric_cols:
        df[col] = df[col].astype(str)
    return df

def get_matched_files(pred_dir: Path, gt_dir: Path) -> list[MatchedFile]:
    """Get matched prediction and ground truth files."""
    matched_files = []
    pred_files = list(pred_dir.glob("*.csv"))
    
    for pred_file in pred_files:
        base_name = pred_file.stem
        gt_base_name = f"AS_gt_{base_name.replace('AS_', '')}"
        gt_file = gt_dir / f"{gt_base_name}.csv"
        
        if gt_file.exists():
            pred_df = pd.read_csv(pred_file)
            gt_df = pd.read_csv(gt_file)
            
            pred_df = convert_to_string(pred_df)
            gt_df = convert_to_string(gt_df)
            
            converted_pred_path = pred_file.parent / f"converted_{pred_file.name}"
            converted_gt_path = gt_file.parent / f"converted_{gt_file.name}"
            
            pred_df.to_csv(converted_pred_path, index=False)
            gt_df.to_csv(converted_gt_path, index=False)
            
            matched_files.append(
                MatchedFile(
                    gt_path=str(converted_gt_path),
                    pred_path=str(converted_pred_path),
                    normalized_name=base_name
                )
            )
    
    return matched_files

def main():
    # Define paths
    pred_dir = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/AS/20241408/20250501_083754/processed/checkpoints/ckpt2_validation")
    gt_dir = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/AS/20241408/20250501_083754/ground_truth")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Get matched files
    matched_files = get_matched_files(pred_dir, gt_dir)
    if not matched_files:
        return
    
    # Initialize path manager and confusion matrix manager
    path_manager = PathManager(output_dir, batch="real_data_analysis")
    cm_manager = ConfusionMatrixManager(
        matched_files=matched_files,
        path_manager=path_manager,
        top_n_problem_classes=20,
        confusion_threshold=0.01
    )
    
    # Run analysis pipeline
    results: ConfusionMatrixAnalysisResults = cm_manager.run()
    
    # Print results summary
    print(f"\nResults saved in: {results.checkpoint_dir}")
    print(f"\nAnalysis Summary:")
    print(f"Total files analyzed: {results.total_files_analyzed}")
    print(f"Successful analyses: {results.successful_analyses}")
    print(f"Failed analyses: {results.failed_analyses}")
    
    if results.has_aggregated_results:
        print("\nAggregated Results:")
        for key, path in results.get_aggregated_results().items():
            print(f"- {key}: {path}")
    
    # Print visualization results
    print("\nVisualization Results:")
    for file_name in results.visualization_results:
        if file_name == 'summary':
            continue
            
        viz_paths = results.get_visualization_paths(file_name)
        if viz_paths:
            print(f"\n{file_name}:")
            for key, path in viz_paths.items():
                if path:
                    print(f"- {key}: {path}")
    
    # Print error analysis results
    print("\nError Analysis Results:")
    for file_name in results.error_analysis_results:
        error_paths = results.get_error_analysis_paths(file_name)
        if error_paths:
            print(f"\n{file_name}:")
            for key, path in error_paths.items():
                print(f"- {key}: {path}")

if __name__ == "__main__":
    main() 
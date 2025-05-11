import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.components.confusion_matrix_component import ConfusionMatrixComponent, MatchedFile
from utils.path_manager import PathManager

def create_sample_data(output_dir: Path):
    """Create sample ground truth and prediction CSV files for testing."""
    # Create first sample data
    gt_data1 = {
        'column1': ['A', 'B', 'A', 'C', 'B'],
        'column2': ['X', 'Y', 'X', 'Z', 'Y']
    }
    pred_data1 = {
        'column1': ['A', 'B', 'B', 'C', 'B'],  # One error in column1
        'column2': ['X', 'Y', 'X', 'Y', 'Z']   # Two errors in column2
    }
    
    # Create second sample data with different values
    gt_data2 = {
        'column1': ['D', 'E', 'D', 'F', 'E'],
        'column2': ['M', 'N', 'M', 'O', 'N']
    }
    pred_data2 = {
        'column1': ['D', 'E', 'E', 'F', 'E'],  # One error in column1
        'column2': ['M', 'N', 'M', 'N', 'O']   # Two errors in column2
    }
    
    # Create DataFrames
    gt_df1 = pd.DataFrame(gt_data1)
    pred_df1 = pd.DataFrame(pred_data1)
    gt_df2 = pd.DataFrame(gt_data2)
    pred_df2 = pd.DataFrame(pred_data2)
    
    # Save to files
    gt_path1 = output_dir / "sample_gt_1.csv"
    pred_path1 = output_dir / "sample_pred_1.csv"
    gt_path2 = output_dir / "sample_gt_2.csv"
    pred_path2 = output_dir / "sample_pred_2.csv"
    
    gt_df1.to_csv(gt_path1, index=False)
    pred_df1.to_csv(pred_path1, index=False)
    gt_df2.to_csv(gt_path2, index=False)
    pred_df2.to_csv(pred_path2, index=False)
    
    return [
        (gt_path1, pred_path1),
        (gt_path2, pred_path2)
    ]

def main():
    # Initialize session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize PathManager
    path_manager = PathManager(
        expected_site_code="test_site",
        batch="test_batch"
    )
    
    # Create output directory for sample data
    output_dir = project_root / "test_data" / "confusion_matrix_example"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    file_pairs = create_sample_data(output_dir)
    
    # Create matched file objects
    matched_files = [
        MatchedFile(
            gt_path=gt_path,
            pred_path=pred_path,
            normalized_name=f"sample_data_{i+1}"
        )
        for i, (gt_path, pred_path) in enumerate(file_pairs)
    ]
    
    # Initialize the component
    component = ConfusionMatrixComponent(
        matched_files=matched_files,
        path_manager=path_manager,
        session_id=session_id,
        verbose=True,
        enable_logging=True,
        enable_console=True,
        log_dir=path_manager.get_log_path(session_id, "confusion_matrix")
    )
    
    # Run the component
    results = component.run({})
    
    # Print results
    print("\nConfusion Matrix Analysis Results:")
    print("==================================")
    
    if results.get('confusion_matrix_results'):
        # Print individual file results
        for file_name, file_results in results['confusion_matrix_results'].items():
            if file_name == 'aggregated':
                continue
                
            print(f"\nFile: {file_name}")
            print(f"Status: {file_results['status']}")
            
            if file_results['status'] == 'success':
                print(f"Columns analyzed: {file_results['columns_analyzed']}")
                print(f"Confusion Matrix: {file_results['confusion_matrix_path']}")
                print(f"Metrics: {file_results['metrics_path']}")
                print(f"Classes: {file_results['classes']}")
        
        # Print aggregated results
        if 'aggregated' in results['confusion_matrix_results']:
            agg_results = results['confusion_matrix_results']['aggregated']
            print("\nAggregated Results:")
            print(f"Status: {agg_results['status']}")
            
            if agg_results['status'] == 'success':
                print(f"Confusion Matrix: {agg_results['confusion_matrix_path']}")
                print(f"Metrics: {agg_results['metrics_path']}")
                print(f"Classes: {agg_results['classes']}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    print(f"\nAll results saved in: {results['checkpoint_dir']}")

if __name__ == "__main__":
    main() 
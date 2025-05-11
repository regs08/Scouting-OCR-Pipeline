import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.components.match_gt_and_file_component import MatchGTAndFileComponent
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
    
    # Create directories
    gt_dir = output_dir / "ground_truth"
    pred_dir = output_dir / "raw"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to files with _gt_ in the filename for ground truth
    gt_path1 = gt_dir / "sample_gt_1.csv"
    pred_path1 = pred_dir / "sample_1.csv"
    gt_path2 = gt_dir / "sample_gt_2.csv"
    pred_path2 = pred_dir / "sample_2.csv"
    
    gt_df1.to_csv(gt_path1, index=False)
    pred_df1.to_csv(pred_path1, index=False)
    gt_df2.to_csv(gt_path2, index=False)
    pred_df2.to_csv(pred_path2, index=False)
    
    # Add an unmatched file to test the matching logic
    unmatched_gt = pd.DataFrame({'column1': ['U', 'V', 'W'], 'column2': ['X', 'Y', 'Z']})
    unmatched_pred = pd.DataFrame({'column1': ['P', 'Q', 'R'], 'column2': ['S', 'T', 'U']})
    
    unmatched_gt.to_csv(gt_dir / "unmatched_gt.csv", index=False)
    unmatched_pred.to_csv(pred_dir / "unmatched_pred.csv", index=False)
    
    return output_dir

def main():
    # Initialize session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize PathManager
    path_manager = PathManager(
        expected_site_code="test_site",
        batch="test_batch"
    )
    
    # Create output directory for sample data
    output_dir = project_root / "test_data" / "match_gt_and_file_example"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    data_dir = create_sample_data(output_dir)
    
    # Initialize the component
    component = MatchGTAndFileComponent(
        path_manager=path_manager,
        session_id=session_id,
        verbose=True,
        enable_logging=True,
        enable_console=True,
        log_dir=path_manager.get_log_path(session_id, "match_gt_and_file")
    )
    
    # Prepare input data
    input_data = {
        'path_manager': path_manager,
        'session_id': session_id,
        'compare_dir': data_dir / "raw",  # Use the raw directory for comparison
        'checkpoint_name': 'unmatched_files'
    }
    
    # Run the component
    results = component.run(input_data)
    
    # Print results
    print("\nFile Matching Results:")
    print("=====================")
    
    # Print matched files
    print("\nMatched Files:")
    for matched_file in results['matched_files']:
        print(f"\nNormalized Name: {matched_file.normalized_name}")
        print(f"Ground Truth: {matched_file.gt_path}")
        print(f"Prediction: {matched_file.pred_path}")
    
    # Print unmatched files
    print("\nUnmatched Files:")
    for moved_file in results['unmatched_moved']:
        print(f"\nType: {moved_file['type']}")
        print(f"File: {moved_file['name']}")
        print(f"Moved from: {moved_file['from']}")
        print(f"Moved to: {moved_file['to']}")
    
    print(f"\nUnmatched files directory: {results['unmatched_dir']}")

if __name__ == "__main__":
    main() 
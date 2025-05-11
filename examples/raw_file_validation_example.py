import sys
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.components.raw_file_validation_component import RawFileValidationComponent
from utils.site_data.site_data_base import SiteDataBase

class ExampleSiteData(SiteDataBase):
    """Example site data class for demonstration"""
    
    def __init__(self):
        super().__init__()
        self.expected_columns = ['timestamp', 'value', 'unit']
        self.required_columns = ['timestamp', 'value']
        self.file_pattern = r'data_\d{8}\.csv'
        self.ignored_files = {'.DS_Store', '.git', '.gitignore'}  # Files to ignore
        
    def validate_file(self, file_path: Path) -> bool:
        """Validate a single file"""
        try:
            # Skip ignored system files
            if file_path.name in self.ignored_files:
                print(f"Skipping system file: {file_path}")
                return False
                
            # Check if file exists and has correct extension
            if not file_path.exists() or file_path.suffix.lower() not in self.allowed_extensions:
                print(f"Invalid file extension for {file_path}. Expected one of: {', '.join(self.allowed_extensions)}")
                return False
                
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            if not all(col in df.columns for col in self.required_columns):
                print(f"Missing required columns in {file_path}. Required: {self.required_columns}")
                return False
                
            # Check if timestamp column is valid
            if 'timestamp' in df.columns:
                try:
                    pd.to_datetime(df['timestamp'])
                except:
                    print(f"Invalid timestamp format in {file_path}")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Error validating file {file_path}: {str(e)}")
            return False

def main():
    # Create example directories
    base_dir = Path("example_data")
    input_dir = base_dir / "data"
    gt_dir = base_dir / "ground_truth"
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example CSV files
    # Input data
    input_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='H'),
        'value': [1.0, 2.0, 3.0, 4.0, 5.0],
        'unit': ['m', 'm', 'm', 'm', 'm']
    })
    input_data.to_csv(input_dir / "data_20240101.csv", index=False)
    
    # Ground truth data
    gt_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='H'),
        'value': [1.1, 2.1, 3.1, 4.1, 5.1],
        'unit': ['m', 'm', 'm', 'm', 'm']
    })
    gt_data.to_csv(gt_dir / "data_20240101.csv", index=False)
    
    # Create an invalid file (missing required column)
    invalid_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='H'),
        'unit': ['m', 'm', 'm', 'm', 'm']
    })
    invalid_data.to_csv(input_dir / "invalid_data.csv", index=False)
    
    # Initialize the component
    validator = RawFileValidationComponent(
        verbose=True,
        enable_logging=True,
        enable_console=True,
        recursive=False,
        stop_on_error=False
    )
    
    # Create site data instance
    site_data = ExampleSiteData()
    
    # Prepare input data
    input_data = {
        'input_dir': str(base_dir),
        'site_data': site_data
    }
    
    # Run validation
    try:
        # Process before pipeline
        prepared_data = validator.process_before_pipeline(input_data)
        
        # Process after pipeline
        results = validator.process_after_pipeline(prepared_data)
        
        # Print results
        print("\nValidation Results:")
        print(f"Total files: {results['valid_file_count'] + results['invalid_file_count']}")
        print(f"Valid files: {results['valid_file_count']}")
        print(f"Invalid files: {results['invalid_file_count']}")
        print("\nValid files:")
        for file in results['valid_files']:
            print(f"- {file}")
            
    except Exception as e:
        print(f"Error during validation: {str(e)}")

if __name__ == "__main__":
    main() 
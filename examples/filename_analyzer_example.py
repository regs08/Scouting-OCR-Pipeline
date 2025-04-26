#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.site_data.site_data_base import SiteDataBase, FileAnalysisResult


class SimpleAnalyzer(SiteDataBase):
    """Simple implementation of SiteDataBase for filename analysis"""
    
    def __init__(self):
        """Initialize with minimal required configuration"""
        super().__init__(
            data_cols=['data1', 'data2'],
            index_cols=['row', 'col'],
            location_pattern=r'([A-Z])(\d+)',  # Generic location pattern
            site_name="filename_analyzer",
            site_code="TEST"
        )
    
    def get_data_column_indices(self):
        """Simple implementation"""
        return [0, 1]
    
    def get_index_column_indices(self):
        """Simple implementation"""
        return [0, 1]


def main():
    """Example of using the filename analyzer"""
    # Create the analyzer
    analyzer = SimpleAnalyzer()
    
    # Example filenames to test
    test_filenames = [
        "SITE1_20230415_R5_P2.jpg",       # Valid
        "DML_20240101_R12_P4.png",        # Valid
        "XYZ123_20211231_A1_B2.pdf",      # Valid
        "TEST_invalid_R1_P1.jpg",         # Invalid date
        "20230101_R1_P1.jpg",             # Missing site code
        "SITE_20230101_R.jpg",            # Invalid location format
        "SITE_20230101.jpg",              # Missing location
    ]
    
    print("Filename Analysis Results:")
    print("=========================")
    
    for filename in test_filenames:
        print(f"\nAnalyzing: {filename}")
        result = analyzer.analyze_filename(filename)
        
        if result.is_valid:
            print(f"  Valid: YES")
            print(f"  Site Code: {result.site_code}")
            print(f"  Date: {result.date}")
            print(f"  Location ID: {result.location_id}")
            print(f"  Row: {result.row}")
            print(f"  Panel: {result.panel}")
        else:
            print(f"  Valid: NO")
            print(f"  Error: {result.error_message}")
            if result.site_code:
                print(f"  Site Code: {result.site_code}")


if __name__ == "__main__":
    main() 
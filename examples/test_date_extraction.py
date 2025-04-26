#!/usr/bin/env python3

import sys
from pathlib import Path
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.site_data.site_data_base import SiteDataBase, FileAnalysisResult


class SimpleExtractor(SiteDataBase):
    """Simple implementation for testing"""
    def __init__(self):
        super().__init__(
            data_cols=['test_col1', 'test_col2'],
            index_cols=['index1'],
            location_pattern=r'R(\d+)P(\d+)',
            site_name="test_site",
            site_code="TEST"
        )
    
    def get_data_column_indices(self):
        return [0, 1]
    
    def get_index_column_indices(self):
        return [0]


def test_filename_date_extraction():
    """Test the date extraction from various filename formats"""
    extractor = SimpleExtractor()
    
    # Test filenames with specified dates
    test_files = [
        # Your specific date format (YYYYMMDD)
        "TEST_20242408_R1P2.jpg",          # Regular format with your date
        "DML_20242408_R5_P10.png",         # With underscores in location part
        "ABC_20242408_XYZ.pdf",            # With arbitrary location
        "TEST_20242408_extra_info_R3P4.jpg",  # With additional text
        
        # Normal date formats
        "TEST_20240824_R1P2.jpg",          # Regular date
        
        # Invalid formats
        "TEST_202408_R1P2.jpg",            # Invalid date (too short)
        "TEST_INVALID_R1P2.jpg",           # Non-numeric date
        "TEST.jpg",                         # No date at all
    ]
    
    print("Testing date extraction from filenames:")
    print("=======================================")
    
    for filename in test_files:
        print(f"\nAnalyzing: {filename}")
        result = extractor.analyze_filename(filename)
        
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
    test_filename_date_extraction() 
#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime

def test_date_validation():
    """Test if various date strings are considered valid by datetime.strptime"""
    test_dates = [
        "20242408",  # Your specific date (YYYYMMDD)
        "20240824",  # Standard format (YYYYMMDD)
        "20241224",  # December 24, 2024
        "20240229",  # Leap year date
        "20230229",  # Invalid date (not a leap year)
        "20240000",  # Invalid date
        "20240133",  # Invalid date
    ]
    
    print("Testing date validation:")
    print("=======================")
    
    for date_str in test_dates:
        print(f"\nTesting date: {date_str}")
        try:
            parsed_date = datetime.strptime(date_str, '%Y%m%d')
            print(f"  Valid: YES")
            print(f"  Parsed as: {parsed_date.strftime('%Y-%m-%d')}")
        except ValueError as e:
            print(f"  Valid: NO")
            print(f"  Error: {str(e)}")
            
    # Test direct parsing of 20242408
    print("\nSpecial test for date 20242408:")
    try:
        year = int("2024")
        month = int("24")
        day = int("08")
        print(f"  Year: {year}, Month: {month}, Day: {day}")
        date = datetime(year, month, day)
        print(f"  Valid: YES")
        print(f"  Date: {date}")
    except ValueError as e:
        print(f"  Valid: NO")
        print(f"  Error: {str(e)}")


if __name__ == "__main__":
    test_date_validation() 
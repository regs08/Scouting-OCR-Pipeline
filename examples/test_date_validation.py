#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Store results for visualization
    results = []
    parsed_dates = []
    
    print("Testing date validation:")
    print("=======================")
    
    for date_str in test_dates:
        print(f"\nTesting date: {date_str}")
        try:
            parsed_date = datetime.strptime(date_str, '%Y%m%d')
            print(f"  Valid: YES")
            print(f"  Parsed as: {parsed_date.strftime('%Y-%m-%d')}")
            results.append(True)
            parsed_dates.append(parsed_date)
        except ValueError as e:
            print(f"  Valid: NO")
            print(f"  Error: {str(e)}")
            results.append(False)
            parsed_dates.append(None)
    
    # Create visualizations
    create_validation_chart(test_dates, results)
    create_date_distribution(parsed_dates)
    
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

def create_validation_chart(dates, results):
    """Create a bar chart showing valid vs invalid dates"""
    plt.figure(figsize=(10, 6))
    colors = ['green' if r else 'red' for r in results]
    plt.bar(dates, [1] * len(dates), color=colors)
    plt.title('Date Validation Results')
    plt.xlabel('Date Strings')
    plt.ylabel('Validation Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('date_validation_results.png')
    plt.close()

def create_date_distribution(parsed_dates):
    """Create a scatter plot of valid dates"""
    valid_dates = [d for d in parsed_dates if d is not None]
    if valid_dates:
        plt.figure(figsize=(10, 6))
        dates = [d.strftime('%Y-%m-%d') for d in valid_dates]
        plt.scatter(range(len(dates)), [1] * len(dates), c='blue', s=100)
        plt.title('Distribution of Valid Dates')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.xticks(range(len(dates)), dates, rotation=45)
        plt.tight_layout()
        plt.savefig('date_distribution.png')
        plt.close()

if __name__ == "__main__":
    test_date_validation() 
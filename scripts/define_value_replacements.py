"""
Script to define value replacements for OCR data cleaning.
This script provides a centralized place to define common value replacements
that need to be standardized in OCR-processed data.
"""

# Common value replacements for OCR data
VALUE_REPLACEMENTS = {
    # Empty or missing values
    "NO VINES": "0",
    "NO VINES,": "0",
    "NO VINES\n": "0",
    "NO VINES\r\n": "0",
    "NO VINES ": "0",
    " NO VINES": "0",
    " NO VINES ": "0"
}

# Numeric columns that should be processed
NUMERIC_COLUMNS = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10",
    "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20"
]

def get_value_replacements() -> dict:
    """
    Get the dictionary of value replacements.
    
    Returns:
        Dictionary mapping values to their replacements
    """
    return VALUE_REPLACEMENTS.copy()

def get_numeric_columns() -> list:
    """
    Get the list of numeric columns to process.
    
    Returns:
        List of column names
    """
    return NUMERIC_COLUMNS.copy()

if __name__ == "__main__":
    # Print the defined replacements and columns
    print("Value Replacements:")
    for old, new in VALUE_REPLACEMENTS.items():
        print(f"  '{old}' → '{new}'")
        
    print("\nNumeric Columns:")
    print("  " + ", ".join(NUMERIC_COLUMNS)) 
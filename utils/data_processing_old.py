import pandas as pd
# Ground Truth Columns
data_cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20']
index_cols = ['date', 'row', 'panel', 'disease']

def clean_cell(val):
    """Clean cell values by converting unwanted values to 0."""
    if isinstance(val, str):
        val_clean = val.strip().lower()
        # Sample Cleaning
        if val_clean in [":selected:", ":unselected:", "/", "1"]:
            return 0
        try:
            float_val = float(val_clean)
            int_val = int(float_val)
            return 0 if int_val == 1 else int_val
        except ValueError:
            return 0
    elif isinstance(val, (int, float)):
        int_val = int(val)
        return 0 if int_val == 1 else int_val
    return val

def clean_dataframe(df, exclude_cols=None):
    """Clean the DataFrame by applying cleaning rules to specified columns."""
    if exclude_cols is None:
        exclude_cols = ["date", "disease", 'panel']
    
    # Create a lowercase version of the column names for matching
    cols_to_clean = [col for col in df.columns if str(col).strip().lower() not in exclude_cols]
    
    # Create a copy of the DataFrame
    df_cleaned = df.copy()
    
    # Apply cleaning only to selected columns
    for col in cols_to_clean:
        df_cleaned[col] = df_cleaned[col].apply(clean_cell)
    
    return df_cleaned 
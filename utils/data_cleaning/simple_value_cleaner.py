import pandas as pd
from typing import List, Optional
from utils.base_processor import BaseProcessor

class SimpleValueCleaner(BaseProcessor):
    """
    A simple data cleaner that replaces empty values and ':selected:' with 0.
    Useful for standardizing values in OCR-processed data.
    """
    
    def __init__(
        self, 
        verbose: bool = False,
        enable_logging: bool = False
    ):
        """
        Initialize the SimpleValueCleaner.
        
        Args:
            verbose: Whether to display detailed output
            enable_logging: Whether to enable logging
        """
        super().__init__(verbose=verbose, enable_logging=enable_logging)
    
    def clean_data(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Replace empty values and ':selected:' with '0' in the specified columns.
        
        Args:
            df: DataFrame to clean
            columns: List of columns to process. If None, processes all columns.
            
        Returns:
            Cleaned DataFrame with standardized values
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # If no columns specified, use all columns
        if columns is None:
            columns = cleaned_df.columns.tolist()
        else:
            # Filter to only include columns that actually exist in the DataFrame
            columns = [col for col in columns if col in cleaned_df.columns]
        
        self.display_and_log(f"Cleaning values in {len(columns)} columns")
        
        # Track replacements for reporting
        replacements = {
            "empty_to_zero": 0,
            "selected_to_zero": 0
        }
        
        # Process each column
        for col in columns:
            # Replace empty values with "0"
            empty_mask = cleaned_df[col].isna() | (cleaned_df[col].astype(str).str.strip() == "")
            replacements["empty_to_zero"] += empty_mask.sum()
            cleaned_df.loc[empty_mask, col] = "0"
            
            # Replace ":selected:" with "0"
            selected_mask = cleaned_df[col].astype(str) == ":selected:"
            replacements["selected_to_zero"] += selected_mask.sum()
            cleaned_df.loc[selected_mask, col] = "0"
        
        # Log summary of replacements
        self.display_and_log("Value cleaning completed", {
            "Empty values replaced": replacements["empty_to_zero"],
            "':selected:' values replaced": replacements["selected_to_zero"],
            "Total replacements": sum(replacements.values())
        })
        
        return cleaned_df
        
    def generate_comparison(
        self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame,
        title: str = "Data Cleaning Comparison",
        max_rows: int = 10,
        max_cols: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate a visual comparison between original and cleaned DataFrames.
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: Cleaned DataFrame after processing
            title: Title for the comparison visualization
            max_rows: Maximum number of rows to display
            max_cols: Maximum number of columns to display
            save_path: Path to save the visualization image
            
        Returns:
            None, but saves a visualization if save_path is provided
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.colors import LinearSegmentedColormap
            
            # Set the style for better appearance
            sns.set_style("whitegrid")
            
            # Get a subset of columns and rows for visualization (to keep it manageable)
            num_rows = min(max_rows, original_df.shape[0])
            num_cols = min(max_cols, original_df.shape[1])
            
            sample_cols = original_df.columns[:num_cols]
            
            # Original unclean data
            unclean_sample = original_df.iloc[:num_rows][sample_cols].copy()
            # Cleaned data
            clean_sample = cleaned_df.iloc[:num_rows][sample_cols].copy()
            
            # Create a DataFrame showing where changes were made
            changes = pd.DataFrame('', index=unclean_sample.index, columns=unclean_sample.columns)
            
            for col in sample_cols:
                for idx in unclean_sample.index[:num_rows]:
                    unclean_val = str(unclean_sample.loc[idx, col])
                    clean_val = str(clean_sample.loc[idx, col])
                    
                    if unclean_val != clean_val:
                        changes.loc[idx, col] = f"{unclean_val} → {clean_val}"
            
            # Create a mask for highlighting changed values
            highlight_mask = changes != ''
            
            # Plot the comparison
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Custom colormap for highlighting changes
            colors = ['#ffffff', '#ffeb99', '#ffd24d']  # White to yellow gradient
            change_cmap = LinearSegmentedColormap.from_list('change_cmap', colors)
            
            # Plot unclean data
            axes[0].set_title(f'Original Data Sample', fontsize=14, fontweight='bold')
            unclean_table = sns.heatmap(
                unclean_sample.astype(str).replace('nan', '').fillna('').replace(':selected:', ':SEL:'),
                annot=True, fmt='', cmap='Greys', cbar=False, linewidths=.5, ax=axes[0]
            )
            
            # Plot cleaned data with highlighting
            axes[1].set_title(f'Cleaned Data Sample', fontsize=14, fontweight='bold')
            clean_table = sns.heatmap(
                clean_sample.astype(str), 
                annot=True, fmt='', 
                cmap='Greys',
                mask=highlight_mask,  # Apply to non-highlighted cells
                cbar=False, linewidths=.5, 
                ax=axes[1]
            )
            
            # Overlay highlighted changes
            highlighted = sns.heatmap(
                clean_sample.astype(str), 
                annot=True, fmt='', 
                cmap=change_cmap,
                mask=~highlight_mask, 
                cbar=False, linewidths=.5, 
                ax=axes[1]
            )
            
            # Set super title
            plt.suptitle(title, fontsize=16, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for super title
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.display_and_log(f"Comparison visualization saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            self.display_and_log(f"Error creating comparison visualization: {str(e)}")
            return None 
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.azure_config import AZURE_FORM_RECOGNIZER_ENDPOINT, AZURE_FORM_RECOGNIZER_KEY
from utils.ocr_processor import OCRProcessor
from collections import defaultdict, namedtuple
from utils.data_preprocessors import DataLoader, DimensionComparison
from utils.col_idx_processing import ArgetSinger24ColIdxProcessor
from utils.file_search.file_matcher import FileMatcher
from utils.base_processor import BaseProcessor
from utils.confusion_matrix_processor import ConfusionMatrixProcessor
from utils.data_cleaning.basic_cleaner import BasicCleaner
from utils.data_cleaning.simple_value_cleaner import SimpleValueCleaner
# Load environment variables
load_dotenv()

# Get the project root directory
root_dir = Path(__file__).parent
data_dir = root_dir / "data"
output_dir = root_dir / "output"
output_dir.mkdir(exist_ok=True)

# Define paths
BASE_DIR = root_dir
DATA_DIR = data_dir
GT_DIR = DATA_DIR / "ground_truth"

# Define named tuples for better readability
FileMatch = namedtuple('FileMatch', ['gt_path', 'pred_path'])
NamedDF = namedtuple('NamedDF', ['gt_df', 'pred_df', 'gt_path', 'pred_path'])


def main():
    """Main entry point for the OCR pipeline."""
    ####
    # Initialize processors
    ####
    data_loader = DataLoader(verbose=False)
    column_processor = DimensionComparison(verbose=True)
    ocr_processor = OCRProcessor()
    static_col_processor = ArgetSinger24ColIdxProcessor(verbose=True)
    ####
    # Define the paths
    ####    
    working_folder = "argetsinger"
    ocr_dir = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ocr_predictions/")
    ocr_output_dir = os.path.join(ocr_dir, working_folder)

    gt_dir = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ground_truth/")
    gt_working_dir = os.path.join(gt_dir, working_folder)
    gt_path = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ground_truth/ground_truth.csv")

    col_matched_dir = "/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/col_matched_ckpt"
    # Create a new directory for the cleaned output
    cleaned_dir = "/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/cleaned_data"
    os.makedirs(cleaned_dir, exist_ok=True)
    
    ####    
    #Process the image with OCR
    ####
    # Uncomment and modify path to process a new image with OCR
    # image_dir = Path("/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/orginal/argetsinger")
    # try:
    #     tables = ocr_processor.process_document(image_dir, save=True, output_dir=ocr_output_dir)
    #     print(f"Successfully processed {len(tables)} tables from image")
      
    # except Exception as e:
    #     print(f"Error processing image with OCR: {str(e)}")
    ####
    # Use FileMatcher to find matching files
    ####
    file_matcher = FileMatcher(working_folder, gt_dir, ocr_dir)
    file_matcher.find_files()
    matched_files = file_matcher.match_files()
    ####
    # Check for 
    #  Loop through matched files and compare using DimensionComparison. Build a list of the results.
    ####
    same_dimensions = {}
    different_dimensions = {}
    for unique_id, files in matched_files.items():
        file_match = FileMatch(gt_path=files['gt_path'], pred_path=files['pred_path'])
        
        # Load the DataFrames
        # reseting the index of the gt dataframe
        gt_df = data_loader.load_df(file_match.gt_path, reset_index=False)
        pred_df = data_loader.load_df(file_match.pred_path, reset_index=True)
        
        # # Perform the comparison
        # #TODO standardize the message output so we can log it  
        comparison_results = column_processor.compare_dimensions(gt_df, pred_df)
        # ####
        # # find the dimensions that match and ditch the rest 
        # ####
        if comparison_results[0] == True:
            same_dimensions[unique_id] = NamedDF(gt_df=gt_df, pred_df=pred_df, 
                                                      gt_path=file_match.gt_path, 
                                                      pred_path=file_match.pred_path)
        else:
            different_dimensions[unique_id] = file_match
    ####
    #replace any columns that don't match. 
    ####
    matching_cols = {}
    unmatched_cols = {}
    for unique_id, match in same_dimensions.items():
        current_gt_df = match.gt_df.copy()
        current_pred_df = match.pred_df.copy()
        
        # Set dataframes and rename columns by index
        static_col_processor.set_dataframes(current_gt_df, current_pred_df)
        static_col_processor.rename_columns_by_index()
        
        # Validate the renamed columns
        results = static_col_processor.validate_dataframes()
        if results[0] == False:
            print(f"Validation failed for {unique_id}")
            print(results[1])
            print(results[2])
            unmatched_cols[unique_id] = match
        else:
            print(f"Validation passed for {unique_id}") 
            col_matched_df = NamedDF(
                gt_df=static_col_processor.gt_df, 
                pred_df=static_col_processor.pred_df, 
                gt_path=match.gt_path, 
                pred_path=match.pred_path
            )
            matching_cols[unique_id] = col_matched_df
            
            # Save validated and renamed files
            print(f"Saving {unique_id} to {col_matched_dir}")
            static_col_processor.save_to_csv(static_col_processor.gt_df, match.gt_path, col_matched_dir)
            static_col_processor.save_to_csv(static_col_processor.pred_df, match.pred_path, col_matched_dir)
    
    ####
    # Clean the data using BasicCleaner
    ####
    print("\n=== Cleaning Data ===")
    basic_cleaner = BasicCleaner(verbose=True, enable_logging=True)
    cleaned_dfs = {}
    
    # Process each matched file pair with the cleaner
    for unique_id, match in matching_cols.items():
        print(f"\nCleaning data for {unique_id}")
        
        # Clean the predicted dataframe (no need to clean ground truth)
        cleaned_pred_df = basic_cleaner.clean_zeros(match.pred_df)
        
        # Store the cleaned data
        cleaned_match = NamedDF(
            gt_df=match.gt_df,
            pred_df=cleaned_pred_df,
            gt_path=match.gt_path,
            pred_path=match.pred_path
        )
        cleaned_dfs[unique_id] = cleaned_match
        
        # Save the cleaned files
        cleaned_file_path = os.path.join(cleaned_dir, f"cleaned_{os.path.basename(match.pred_path)}")
        cleaned_pred_df.to_csv(cleaned_file_path, index=False)
        print(f"Saved cleaned file to {cleaned_file_path}")
    
    ####
    # Example of using SimpleValueCleaner to clean data
    ####
    print("\n=== Cleaning Data with SimpleValueCleaner ===")
    simple_cleaner = SimpleValueCleaner(verbose=True, enable_logging=True)
    simple_cleaned_dfs = {}
    
    # Process each matched file pair with the SimpleValueCleaner
    for unique_id, match in matching_cols.items():
        print(f"\nSimple cleaning data for {unique_id}")
        
        # Save the unclean version first for comparison
        unclean_file_path = os.path.join(cleaned_dir, f"unclean_{os.path.basename(match.pred_path)}")
        match.pred_df.to_csv(unclean_file_path, index=False)
        print(f"Saved unclean file to {unclean_file_path}")
        
        # Clean the predicted dataframe using the new SimpleValueCleaner
        simple_cleaned_pred_df = simple_cleaner.clean_data(match.pred_df)
        
        # Store the cleaned data
        simple_cleaned_match = NamedDF(
            gt_df=match.gt_df,
            pred_df=simple_cleaned_pred_df,
            gt_path=match.gt_path,
            pred_path=match.pred_path
        )
        simple_cleaned_dfs[unique_id] = simple_cleaned_match
        
        # Save the cleaned files
        simple_cleaned_file_path = os.path.join(cleaned_dir, f"simple_cleaned_{os.path.basename(match.pred_path)}")
        simple_cleaned_pred_df.to_csv(simple_cleaned_file_path, index=False)
        print(f"Saved simple cleaned file to {simple_cleaned_file_path}")
        
        # Create visual comparison between unclean and cleaned data
        try:
            comparison_path = os.path.join(cleaned_dir, f"comparison_{unique_id}.png")
            simple_cleaner.generate_comparison(
                original_df=match.pred_df,
                cleaned_df=simple_cleaned_pred_df,
                title=f"Data Cleaning Comparison - {unique_id}",
                save_path=comparison_path
            )
            print(f"Generated visual comparison: {comparison_path}")
        except Exception as vis_error:
            print(f"Warning: Could not create visual comparison: {str(vis_error)}")
    
    # Create a summary visualization of cleaning statistics
    try:
        print("\nGenerating cleaning summary visualization...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Collect statistics about cleaning for each file
        cleaning_stats = []
        
        for unique_id, match in simple_cleaned_dfs.items():
            # Original unclean dataframe
            unclean_df = matching_cols[unique_id].pred_df
            # Cleaned dataframe
            clean_df = match.pred_df
            
            # Count different types of cleanings
            empty_count = 0
            selected_count = 0
            
            # Check each cell for what was changed
            for col in unclean_df.columns:
                for idx in unclean_df.index:
                    unclean_val = str(unclean_df.loc[idx, col])
                    
                    # Check for empty values
                    if pd.isna(unclean_df.loc[idx, col]) or unclean_val.strip() == "":
                        empty_count += 1
                    
                    # Check for :selected: values
                    elif unclean_val == ":selected:":
                        selected_count += 1
            
            # Calculate percentage of cells that were cleaned
            total_cells = unclean_df.size
            total_cleaned = empty_count + selected_count
            percent_cleaned = (total_cleaned / total_cells) * 100 if total_cells > 0 else 0
            
            # Add to statistics
            cleaning_stats.append({
                'Document': unique_id,
                'Total Cells': total_cells,
                'Empty Values': empty_count,
                'Selected Values': selected_count,
                'Total Cleaned': total_cleaned,
                'Percent Cleaned': percent_cleaned
            })
        
        # Create a DataFrame with the statistics
        stats_df = pd.DataFrame(cleaning_stats)
        
        # Only proceed if we have data
        if not stats_df.empty:
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 1. Bar chart showing counts of each type of cleaning per document
            bar_data = stats_df.set_index('Document')[['Empty Values', 'Selected Values']]
            bar_data.plot(kind='bar', stacked=True, ax=axes[0], 
                        color=['#3498db', '#e74c3c'])
            axes[0].set_title('Number of Values Cleaned by Type', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_xlabel('Document', fontsize=12)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            axes[0].legend(title='Cleaning Type')
            
            # Add value labels on top of bars
            for i, doc in enumerate(bar_data.index):
                total = bar_data.loc[doc, 'Empty Values'] + bar_data.loc[doc, 'Selected Values']
                axes[0].text(i, total + 5, f'{total}', ha='center', fontweight='bold')
            
            # 2. Pie chart showing overall distribution of cleaning types
            total_empty = stats_df['Empty Values'].sum()
            total_selected = stats_df['Selected Values'].sum()
            total_untouched = stats_df['Total Cells'].sum() - (total_empty + total_selected)
            
            pie_data = [total_empty, total_selected, total_untouched]
            pie_labels = ['Empty Values', ':selected: Values', 'Unchanged']
            pie_colors = ['#3498db', '#e74c3c', '#95a5a6']
            
            axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                       startangle=90, colors=pie_colors, explode=(0.05, 0.05, 0))
            axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            axes[1].set_title('Overall Distribution of Cleaning Operations', fontsize=14, fontweight='bold')
            
            # Add a text box with summary statistics
            total_docs = len(stats_df)
            total_cells_all = stats_df['Total Cells'].sum()
            total_cleaned_all = stats_df['Total Cleaned'].sum()
            percent_cleaned_all = (total_cleaned_all / total_cells_all) * 100 if total_cells_all > 0 else 0
            
            stats_text = (
                f"Summary Statistics:\n"
                f"Documents: {total_docs}\n"
                f"Total Cells: {total_cells_all}\n"
                f"Cells Cleaned: {total_cleaned_all} ({percent_cleaned_all:.1f}%)\n"
                f"Empty Values: {total_empty}\n"
                f":selected: Values: {total_selected}"
            )
            
            # Place text box in bottom right of the figure
            plt.figtext(0.9, 0.02, stats_text, ha='right', bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
            
            # Adjust layout
            plt.suptitle('Data Cleaning Summary', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the suptitle
            
            # Save the summary visualization
            summary_path = os.path.join(cleaned_dir, "cleaning_summary.png")
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Generated cleaning summary visualization: {summary_path}")
            
            # Also save the statistics as a CSV
            stats_csv_path = os.path.join(cleaned_dir, "cleaning_statistics.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"Saved cleaning statistics to: {stats_csv_path}")
            
    except Exception as e:
        print(f"Warning: Could not create cleaning summary visualization: {str(e)}")
    
    ####
    #Then we make our confusion matrix for each comparison using one-vs-all approach
    ####
    
    # Initialize the ConfusionMatrixProcessor
    confusion_processor = ConfusionMatrixProcessor(
        case_sensitive=False,
        output_dir=cleaned_dir,  # Change output directory to cleaned_dir
        verbose=True,
        enable_logging=True
    )
    
    # Store analysis results for each file comparison
    analysis_results = {}
    # Store all confusion matrices for aggregation
    all_confusion_matrices = []
    
    # Process each matched file pair
    for unique_id, match in cleaned_dfs.items():  # Use cleaned data instead of original
        try:
            print(f"\n=== One-vs-All Analysis for {unique_id} (Cleaned) ===")
            
            # Check if both files exist
            if match.gt_df is None or match.pred_df is None:
                print(f"Skipping {unique_id}: One or both DataFrames are missing")
                continue
                
            # Perform one-vs-all analysis using the confusion matrix processor
            results = confusion_processor.one_vs_all_analysis(
                gt_df=match.gt_df,
                pred_df=match.pred_df,
                columns=match.gt_df.columns,
                unique_id=f"cleaned_{unique_id}"
            )
            
            # Store results for aggregation
            analysis_results[unique_id] = results
            
            # Create and save a multiclass confusion matrix
            multi_cm = confusion_processor.create_multiclass_confusion_matrix(
                gt_df=match.gt_df,
                pred_df=match.pred_df,
                columns=match.gt_df.columns
            )
            
            # Store the confusion matrix for later aggregation
            all_confusion_matrices.append(multi_cm)
            
            # Save visualization of the confusion matrix
            try:
                vis_output_path = os.path.join(cleaned_dir, f"cleaned_{unique_id}_confusion_matrix.png")
                confusion_processor.visualize_confusion_matrix(
                    confusion_matrix=multi_cm,
                    title=f"Confusion Matrix - {unique_id} (Cleaned)",
                    save_path=vis_output_path
                )
                print(f"Confusion matrix visualization saved to: {vis_output_path}")
            except Exception as vis_error:
                print(f"Warning: Could not create visualization: {str(vis_error)}")
            
            # Print summary results
            print("\nPer-Value Performance (After Cleaning):")
            pd.set_option('display.max_colwidth', None)
            print(results['summary'].to_string(index=False))
            
            # Print problematic values (low F1 score)
            poor_performance = results['summary'][results['summary']['F1 Score'].str.rstrip('%').astype(float) < 80]
            if not poor_performance.empty:
                print("\nValues with Poor Performance (F1 < 80%):")
                print(poor_performance.to_string(index=False))
                
                # Print detailed confusion patterns for poor performing values
                print("\nDetailed Confusion Patterns for Poor Performing Values:")
                for _, row in poor_performance.iterrows():
                    value = row['Value']
                    print(f"\nGround Truth Value: '{value}'")
                    print("When this was the correct value, it was predicted as:")
                    confusions = results['confusion_patterns'].get(value, {})
                    if confusions:
                        for pred_val, count in sorted(confusions.items(), key=lambda x: x[1], reverse=True):
                            print(f"  '{pred_val}': {count} times")
                    else:
                        print("  No confusion patterns found")
                        
                    print(f"\nWhen '{value}' was predicted, the actual ground truth was:")
                    reverse_conf = results['reverse_patterns'].get(value, {})
                    if reverse_conf:
                        for gt_val, count in sorted(reverse_conf.items(), key=lambda x: x[1], reverse=True):
                            print(f"  '{gt_val}': {count} times")
                    else:
                        print("  No incorrect predictions found")
                
        except Exception as e:
            print(f"Error analyzing {unique_id}: {str(e)}")
            continue
    
    ####
    # Create aggregated confusion matrix from all individual matrices
    ####
    if all_confusion_matrices:
        try:
            print("\n=== Creating Aggregated Confusion Matrix (Cleaned Data) ===")
            
            # Define the output path for the aggregated visualization
            aggregated_output_path = os.path.join(cleaned_dir, "cleaned_aggregated_confusion_matrix.png")
            
            # Aggregate all the confusion matrices
            aggregated_matrix = confusion_processor.aggregate_confusion_matrices(
                confusion_matrices=all_confusion_matrices,
                title="Aggregated Confusion Matrix - All Documents (Cleaned)",
                save_path=aggregated_output_path,
                max_display_size=30  # Limit to top 30 values for better visibility
            )
            
            print(f"Aggregated confusion matrix created with shape {aggregated_matrix.shape}")
            print(f"Visualization saved to: {aggregated_output_path}")
            print(f"Full matrix saved to: {aggregated_output_path.replace('.png', '_full.csv')}")
            
        except Exception as e:
            print(f"Error creating aggregated confusion matrix: {str(e)}")
    
    ####
    #Then we aggregate the results across all files
    ####
    if analysis_results:
        try:
            print("\n=== Overall Analysis Results (Cleaned Data) ===")
            
            # Aggregate results across all files
            overall_df = confusion_processor.aggregate_results(analysis_results)
            
            print("\nOverall Per-Value Performance (After Cleaning):")
            print(overall_df.to_string(index=False))
            
            # Create a visualization of the top N values by F1 score
            try:
                # Get top 20 values for visualization
                top_values = overall_df.head(20).copy()
                
                # Convert percentage strings to float values for plotting
                for col in ['Precision', 'Recall', 'F1 Score']:
                    top_values[col] = top_values[col].str.rstrip('%').astype(float) / 100
                
                # Create a bar chart of the top values
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Set the style for better appearance
                sns.set_style("whitegrid")
                
                # Create a larger figure for better spacing
                plt.figure(figsize=(16, 10))
                
                # Create a grouped bar chart
                x = range(len(top_values))
                width = 0.25
                
                # Add some padding between bars
                plt.bar([i - width*1.1 for i in x], top_values['Precision'], width=width, 
                        label='Precision', color='#4285F4', alpha=0.8)
                plt.bar(x, top_values['Recall'], width=width, 
                        label='Recall', color='#34A853', alpha=0.8)
                plt.bar([i + width*1.1 for i in x], top_values['F1 Score'], width=width, 
                        label='F1 Score', color='#EA4335', alpha=0.8)
                
                # Customize the plot for better readability
                plt.xlabel('Value', fontsize=12, fontweight='bold')
                plt.ylabel('Score', fontsize=12, fontweight='bold')
                plt.title('Top 20 Values by F1 Score (Cleaned Data)', fontsize=16, fontweight='bold')
                
                # Add some padding to x-axis
                plt.xlim(-0.5, len(top_values) - 0.5)
                
                # Set grid on y-axis only
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add a horizontal line at y=0.8 (80%) to highlight good performance
                plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
                plt.text(-0.5, 0.81, '80% threshold', fontsize=10, color='gray')
                
                # Format the y-axis as percentage
                plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
                
                # Position x-ticks at the center of the groups
                plt.xticks(x, top_values['Value'], rotation=45, ha='right', fontsize=10)
                
                # Add value labels on the bars for F1 score only (to avoid cluttering)
                for i, v in enumerate(top_values['F1 Score']):
                    plt.text(i + width*1.1, v + 0.02, f"{v:.0%}", 
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
                # Place the legend outside of the main plot
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                          fancybox=True, shadow=True, ncol=3, fontsize=10)
                
                # Add more spacing at the bottom for the legend and rotated labels
                plt.subplots_adjust(bottom=0.2)
                
                plt.tight_layout()
                
                # Save the visualization with higher DPI for better quality
                vis_output_path = os.path.join(cleaned_dir, "cleaned_overall_performance.png")
                plt.savefig(vis_output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"\nOverall performance visualization saved to: {vis_output_path}")
            except Exception as vis_error:
                print(f"Warning: Could not create overall visualization: {str(vis_error)}")
            
        except Exception as e:
            print(f"Error in overall analysis: {str(e)}")
    
    ####
    #Then we make handling cases for each of the results.
    ####


if __name__ == "__main__":
    main() 
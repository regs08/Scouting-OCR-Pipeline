from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.azure_config import AZURE_FORM_RECOGNIZER_ENDPOINT, AZURE_FORM_RECOGNIZER_KEY
import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import pandas as pd

from utils.image_handler import ImageHandler
from .base_session_processor import BaseSessionProcessor

class OCRProcessor(BaseSessionProcessor):
    def __init__(self,
                 path_manager,
                 session_id: str,
                 verbose: bool = True,
                 enable_logging: bool = True,
                 enable_console: bool = True,
                 log_dir: Optional[Path] = None,
                 operation_name: Optional[str] = None):
        """Initialize the OCR processor with Azure Form Recognizer client.
        
        Args:
            path_manager: PathManager instance for handling file paths
            session_id: Unique identifier for the session
            verbose: Whether to show detailed output
            enable_logging: Whether to enable logging to file
            enable_console: Whether to enable console output
            log_dir: Directory where log files will be stored
            operation_name: Name of the current operation/checkpoint
        """
        super().__init__(
            path_manager=path_manager,
            session_id=session_id,
            verbose=verbose,
            enable_logging=enable_logging,
            enable_console=enable_console,
            log_dir=log_dir,
            operation_name=operation_name or "ocr_processing"
        )
        
        self.client = DocumentAnalysisClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )
        self.image_handler = ImageHandler()

    def _create_table_data(self, table) -> list:
        """Create a 2D list representation of a table from OCR results.
        
        Args:
            table: Table object from Azure Form Recognizer
            
        Returns:
            2D list containing the table data
        """
        # Initialize empty table with correct dimensions
        table_data = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
        
        # Fill in cell contents
        for cell in table.cells:
            table_data[cell.row_index][cell.column_index] = cell.content
            
        return table_data

    def _save_table_data(self, table_data: list, image_path: Union[str, Path], table_index: int = 0) -> pd.DataFrame:
        """Save OCR table data as DataFrame and CSV.
        
        Args:
            table_data: 2D list containing table data
            image_path: Path to the original image file
            table_index: Index of the table in the document
            
        Returns:
            DataFrame containing the table data
        """
        # Convert table data to DataFrame
        df = pd.DataFrame(table_data)
        
        # Drop the index
        df.reset_index(drop=True, inplace=True)
        
        # Get the original image filename without extension
        image_path = Path(image_path)
        base_name = image_path.stem
        
        # Create output filename with table index if multiple tables
        output_name = f"pred_{base_name}"
        if table_index > 0:
            output_name += f"_table{table_index + 1}"
        
        # Get output directory from path manager
        output_dir = self.path_manager.get_checkpoint_path(
            session_id=self.session_id,
            checkpoint="ckpt1_ocr_processed"
        )
        
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)
        self.log_info("save_table", f"Saved OCR predictions to: {csv_path}")
        return df

    def _extract_tables(self, result, image_path: Union[str, Path]) -> list:
        """Extract tables from OCR results.
        
        Args:
            result: Result object from Azure Form Recognizer
            image_path: Path to the original image file
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        dataframes = []
        for idx, table in enumerate(result.tables):
            self.log_info("extract_tables", f"Processing Table {idx + 1}")
            table_data = self._create_table_data(table)
            tables.append(table_data)
            
            df = self._save_table_data(table_data, image_path, idx)
            dataframes.append(df)
            self.log_info("extract_tables", f"Table {idx + 1} shape: {df.shape}")

        if not tables:
            self.log_warning("extract_tables", "No tables found in the document")
        
        return dataframes

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process images in the session directory using OCR.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Dictionary with processing results
        """
        session_dir = input_data.get('session_dir')
        if not session_dir:
            raise ValueError("Missing session_dir in input data")
            
        session_dir = Path(session_dir)
        
        # Get the raw directory path where images are stored
        raw_dir = self.path_manager.get_session_paths(self.session_id)['raw']
        
        if not raw_dir.exists():
            self.log_error("process", f"Raw directory not found: {raw_dir}")
            raise ValueError(f"Raw directory not found: {raw_dir}")
            
        # Get the output directory
        output_dir = self.path_manager.get_checkpoint_path(self.session_id, "ckpt1_ocr_processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track results
        processed_files = []
        
        # Process each image in the raw directory
        for image_path in raw_dir.glob("**/*.png"):
            try:
                self.log_info("process", f"Processing image: {image_path}")
                
                # Load the image
                image = self.image_handler.load_image(image_path)
                
                # Process the image with Azure Form Recognizer
                poller = self.client.begin_analyze_document("prebuilt-layout", document=image)
                result = poller.result()
                
                # Extract tables from the result
                tables = self._extract_tables(result, image_path)
                
                # Record processing results
                csv_files = []
                for idx, df in enumerate(tables):
                    # Create output filename
                    output_name = f"pred_{image_path.stem}"
                    if idx > 0:
                        output_name += f"_table{idx + 1}"
                        
                    output_path = output_dir / f"{output_name}.csv"
                    csv_files.append(output_path)
                
                processed_files.append({
                    'image_path': str(image_path),
                    'table_count': len(tables),
                    'csv_files': [str(path) for path in csv_files]
                })
                
                self.log_info("process", f"Successfully processed {image_path.name}")
                
            except Exception as e:
                self.log_error("process", f"Error processing {image_path.name}: {str(e)}")
        
        # Return processing results
        return {
            'processed_images': len(processed_files),
            'processed_files': processed_files,
            'ocr_output_dir': str(output_dir)
        } 
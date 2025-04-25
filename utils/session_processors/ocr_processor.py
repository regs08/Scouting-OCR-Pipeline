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
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif'}

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

    def _save_table_data(self, table_data: list, file_path: Union[str, Path], table_index: int = 0) -> pd.DataFrame:
        """Save OCR table data as DataFrame and CSV.
        
        Args:
            table_data: 2D list containing table data
            file_path: Path to the original file
            table_index: Index of the table in the document
            
        Returns:
            DataFrame containing the table data
        """
        # Convert table data to DataFrame
        df = pd.DataFrame(table_data)
        
        # Drop the index
        df.reset_index(drop=True, inplace=True)
        
        # Get the original filename without extension
        file_path = Path(file_path)
        base_name = file_path.stem
        
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

    def _extract_tables(self, result, file_path: Union[str, Path]) -> list:
        """Extract tables from OCR results.
        
        Args:
            result: Result object from Azure Form Recognizer
            file_path: Path to the original file
            
        Returns:
            List of extracted tables as DataFrames
        """
        tables = []
        dataframes = []
        for idx, table in enumerate(result.tables):
            self.log_info("extract_tables", f"Processing Table {idx + 1}")
            table_data = self._create_table_data(table)
            tables.append(table_data)
            
            df = self._save_table_data(table_data, file_path, idx)
            dataframes.append(df)
            self.log_info("extract_tables", f"Table {idx + 1} shape: {df.shape}")

        if not tables:
            self.log_warning("extract_tables", "No tables found in the document")
        
        return dataframes

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if the file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Boolean indicating if the file format is supported
        """
        return file_path.suffix.lower() in self.supported_formats

    def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file using Azure Form Recognizer.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing processing results
        """
        try:
            self.log_info("process_file", f"Processing file: {file_path}")
            
            if not self._is_supported_file(file_path):
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # For PDFs and images, we can send the file directly to Form Recognizer
            with open(file_path, "rb") as file:
                poller = self.client.begin_analyze_document("prebuilt-layout", document=file)
                result = poller.result()
            
            # Extract tables from the result
            tables = self._extract_tables(result, file_path)
            
            # Record processing results
            csv_files = []
            for idx in range(len(tables)):
                output_name = f"pred_{file_path.stem}"
                if idx > 0:
                    output_name += f"_table{idx + 1}"
                    
                output_path = self.path_manager.get_checkpoint_path(
                    self.session_id, "ckpt1_ocr_processed"
                ) / f"{output_name}.csv"
                csv_files.append(output_path)
            
            return {
                'file_path': str(file_path),
                'table_count': len(tables),
                'csv_files': [str(path) for path in csv_files]
            }
                
        except Exception as e:
            self.log_error("process_file", f"Error processing {file_path.name}: {str(e)}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process files in the session directory using OCR.
        
        Args:
            input_data: Dictionary containing session data
            
        Returns:
            Dictionary with processing results
        """
        session_dir = input_data.get('session_dir')
        if not session_dir:
            raise ValueError("Missing session_dir in input data")
            
        session_dir = Path(session_dir)
        
        # Get the raw directory path where files are stored
        raw_dir = self.path_manager.get_session_paths(self.session_id)['raw']
        
        if not raw_dir.exists():
            self.log_error("process", f"Raw directory not found: {raw_dir}")
            raise ValueError(f"Raw directory not found: {raw_dir}")
            
        # Get the output directory
        output_dir = self.path_manager.get_checkpoint_path(self.session_id, "ckpt1_ocr_processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track results
        processed_files = []
        
        # Process each supported file in the raw directory
        for file_format in self.supported_formats:
            for file_path in raw_dir.glob(f"**/*{file_format}"):
                try:
                    result = self._process_file(file_path)
                    processed_files.append(result)
                    self.log_info("process", f"Successfully processed {file_path.name}")
                except Exception as e:
                    self.log_error("process", f"Error processing {file_path.name}: {str(e)}")
        
        # Return processing results
        return {
            'processed_files': len(processed_files),
            'processed_file_details': processed_files,
            'ocr_output_dir': str(output_dir)
        } 
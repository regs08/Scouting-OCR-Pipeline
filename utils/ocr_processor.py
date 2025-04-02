from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.azure_config import AZURE_FORM_RECOGNIZER_ENDPOINT, AZURE_FORM_RECOGNIZER_KEY
import os
from pathlib import Path
from typing import Union, Optional
import imghdr
import pandas as pd
from PIL import Image

class OCRProcessor:
    def __init__(self):
        """Initialize the Azure Form Recognizer client for OCR processing."""
        self.client = DocumentAnalysisClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )
        
        # Common image formats supported by Azure Form Recognizer
        self.supported_formats = {
            'jpeg', 'jpg', 'png', 'bmp', 'tiff', 'heic', 'heif'
        }

    def validate_image(self, image_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
        """Validate if the image file exists and is in a supported format for OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if the image is valid, False otherwise
            - error_message: Description of the error if invalid, None if valid
        """
        try:
            # Convert to Path object
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                return False, f"Image file not found: {image_path}"
            
            # Check if it's a file (not a directory)
            if not image_path.is_file():
                return False, f"Path is not a file: {image_path}"
            
            # Check file size (Azure Form Recognizer has a 50MB limit)
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                return False, f"File size ({file_size_mb:.2f}MB) exceeds 50MB limit"
            
            # For PNG files, just check the extension
            # Check if file extension is supported
            if image_path.suffix.lower()[1:] not in self.supported_formats:
                return False, f"Unsupported image format: {image_path.suffix}. Supported formats: {', '.join(self.supported_formats)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"

    def load_image(self, image_path: Union[str, Path]) -> bytes:
        """Load and validate an image file for OCR processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image data as bytes
            
        Raises:
            ValueError: If the image is invalid or cannot be loaded
        """
        # Validate the image first
        is_valid, error_message = self.validate_image(image_path)
        if not is_valid:
            raise ValueError(error_message)
        
        try:
            # Read the image file
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Verify the data is not empty
            if not image_data:
                raise ValueError(f"Image file is empty: {image_path}")
            
            return image_data
            
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

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
        output_name = f"{base_name}_ocr_pred"
        if table_index > 0:
            output_name += f"_table{table_index + 1}"
        
        # Create output directory if it doesn't exist
        output_dir = Path("data/ocr_predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved OCR predictions to: {csv_path}")
        return df

    def _extract_tables(self, result, image_path: Union[str, Path]) -> list:
        """Extract tables from OCR results.
        
        Args:
            result: Result object from Azure Form Recognizer
            image_path: Path to the original image file
            
        Returns:
            List of extracted tables as 2D lists
        """
        tables = []
        for idx, table in enumerate(result.tables):
            print(f"\nProcessing Table {idx + 1}")
            table_data = self._create_table_data(table)
            tables.append(table_data)
            
            # Save each table as DataFrame and CSV
            df = self._save_table_data(table_data, image_path, idx)
            print(f"Table {idx + 1} shape: {df.shape}")
        
        if not tables:
            print("Warning: No tables found in the document")
        
        return tables

    def process_document(self, image_path: Union[str, Path]) -> list:
        """Process an image using OCR and return extracted table data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of tables extracted from the document
            
        Raises:
            ValueError: If the image is invalid or cannot be processed
        """
        try:
            # Load and validate the image
            image_data = self.load_image(image_path)
            
            # Analyze document using the prebuilt-layout model
            poller = self.client.begin_analyze_document("prebuilt-layout", document=image_data)
            result = poller.result()
            
            # Extract tables
            return self._extract_tables(result, image_path)
            
        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}") 
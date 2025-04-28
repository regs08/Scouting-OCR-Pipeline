from pathlib import Path
from typing import Dict, Any
import shutil
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.azure_config import AZURE_FORM_RECOGNIZER_ENDPOINT, AZURE_FORM_RECOGNIZER_KEY
import pandas as pd

from utils.pipeline_component import PipelineComponent

class OCRProcessorComponent(PipelineComponent):
    """
    Pipeline component that processes files using Azure Form Recognizer OCR.
    Failed files are moved to the flagged directory under the checkpoint name.
    """
    def __init__(self, **kwargs):
        # Do NOT set operation_name here; let the pipeline infra handle it
        super().__init__(**kwargs)
        self.client = DocumentAnalysisClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif'}
        self.path_manager = kwargs.get('path_manager')

    def process_before_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.path_manager = input_data.get('path_manager')
        self.session_id = input_data.get('session_id')
        if not self.path_manager or not self.session_id:
            raise ValueError("path_manager and session_id are required in input_data")
        
        session_paths = self.path_manager.get_session_paths(self.session_id)
        self.raw_dir = session_paths.get('raw')
        self.flagged_dir = session_paths.get('flagged')
        if not self.raw_dir or not self.flagged_dir:
            raise ValueError("raw and flagged directories must exist in session paths")
        
        # Create error directory under flagged
        self.error_dir = self.flagged_dir / input_data.get('checkpoint_name', 'ocr_errors')
        self.error_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory for OCR results
        self.output_dir = self.path_manager.get_checkpoint_path(
            self.session_id, "ckpt1_ocr_processed"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        return input_data

    def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file with OCR and save results."""
        try:
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Process with Form Recognizer
            with open(file_path, "rb") as file:
                poller = self.client.begin_analyze_document("prebuilt-layout", document=file)
                result = poller.result()

            tables_processed = []
            for idx, table in enumerate(result.tables):
                # --- FIXED: Build table using row_index and column_index ---
                table_data = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
                for cell in table.cells:
                    table_data[cell.row_index][cell.column_index] = cell.content
                df = pd.DataFrame(table_data)
                # ----------------------------------------------------------

                # Create output filename
                output_name = f"{file_path.stem}"
                if idx > 0:
                    output_name += f"_table{idx + 1}"
                output_path = self.output_dir / f"{output_name}.csv"
                
                # Save table
                df.to_csv(output_path, index=False)
                tables_processed.append(str(output_path))

            return {
                'status': 'success',
                'file': str(file_path),
                'tables_processed': len(tables_processed),
                'output_files': tables_processed
            }

        except Exception as e:
            # Move failed file to error directory
            error_file = self.error_dir / file_path.name
            shutil.move(str(file_path), str(error_file))
            
            self.log_error("process_file", f"Failed to process {file_path.name}", {
                'error': str(e),
                'moved_to': str(error_file)
            })
            
            return {
                'status': 'error',
                'file': str(file_path),
                'error': str(e),
                'moved_to': str(error_file)
            }

    def process_after_pipeline(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        processed_files = []
        error_files = []

        # Process each file in raw directory
        for file_path in self.raw_dir.iterdir():
            if file_path.is_file():
                result = self._process_file(file_path)
                if result['status'] == 'success':
                    processed_files.append(result)
                else:
                    error_files.append(result)

        output = {
            **pipeline_output,
            'ocr_processed': {
                'successful': processed_files,
                'failed': error_files,
                'output_dir': str(self.output_dir),
                'error_dir': str(self.error_dir)
            },
            'ocr_output_dir': str(self.output_dir)
        }
        
        return output 
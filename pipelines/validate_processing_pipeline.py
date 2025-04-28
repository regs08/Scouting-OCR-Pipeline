from utils.components.match_gt_and_file_component import MatchGTAndFileComponent
from utils.components.copy_validated_compare_component import CopyValidatedCompareComponent
from utils.managers.validate_processing_manager import ValidateProcessingManager
from utils.runnable_component_config import RunnableComponentConfig 

match_gt_and_file_config = RunnableComponentConfig(
    component_class=MatchGTAndFileComponent,
    checkpoint_name="ckpt1_match_gt_and_processed",
    checkpoint_number=1,
    description="Match GT files to processed files and move unmatched to flagged/unmatched directory",
    metadata={'compare_dir_key': 'processed/checkpoints/ckpt1_ocr_processed'}
)

copy_validted_files_config = RunnableComponentConfig(
    component_class=CopyValidatedCompareComponent,
    checkpoint_name="ckpt2_validation",
    checkpoint_number=2,
    description="Load matched GT and processed CSVs as DataFrames"
)

validate_config = RunnableComponentConfig(
    component_class=ValidateProcessingManager,
    checkpoint_name="ckpt3_validate_processing",
    checkpoint_number=3,
    description="Validate the processing of the files",
    component_configs=[
        match_gt_and_file_config,
        copy_validted_files_config
    ]
)
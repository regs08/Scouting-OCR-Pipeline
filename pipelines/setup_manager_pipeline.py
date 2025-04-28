from utils.components.raw_file_validation_component import RawFileValidationComponent
from utils.components.copy_files_to_session import CopyRawFilesToSessionComponent
from utils.components.gt_file_validation_component import GTFileValidationComponent
from utils.components.match_gt_and_file_component import MatchGTAndFileComponent
from utils.components.directory_creator import DirectoryCreator
from utils.runnable_component_config import RunnableComponentConfig
from utils.managers.setup_manager import SetupManager

# Setup manager pipeline components
# Define directory creator component
directory_setup_config = RunnableComponentConfig(
    component_class=DirectoryCreator,
    checkpoint_name="directory_setup",
    checkpoint_number=1,
    description="Set up directory structure"
)

file_validator_config = RunnableComponentConfig(
    component_class=RawFileValidationComponent,
    checkpoint_name="file_validation",
    checkpoint_number=2,
    description="validate raw files"
)

copy_raw_files_config = RunnableComponentConfig(
    component_class=CopyRawFilesToSessionComponent,
    checkpoint_name="copy_files_to_session",
    checkpoint_number=3,
    description="Copy validated files to session"
)
copy_gt_files_config = RunnableComponentConfig(
    component_class=GTFileValidationComponent,
    checkpoint_name="file_validation",
    checkpoint_number=4,
    description="Copy validate_gt_file to session"
)

match_gt_and_raw_config = RunnableComponentConfig(
    component_class=MatchGTAndFileComponent,
    checkpoint_name="match_gt_and_raw",
    checkpoint_number=5,
    description="Match GT and raw data folders and flag unmatched"
)

setup_config = RunnableComponentConfig(
    component_class=SetupManager,
    checkpoint_name="ckpt1_setup",
    checkpoint_number=1,
    description="Initial setup and data validation",
    metadata={
        "setup_type": "initial",
        "requires_validation": True
    },
    component_configs=[
        directory_setup_config,
        file_validator_config,
        copy_raw_files_config,
        copy_gt_files_config,
        match_gt_and_raw_config
    ]
)

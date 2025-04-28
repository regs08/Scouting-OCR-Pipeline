from utils.runnable_component_config import RunnableComponentConfig
from utils.components.ocr_processor_component import OCRProcessorComponent
from utils.managers.model_manager import ModelManager
# Define model pipeline components
ocr_processor_config = RunnableComponentConfig(
        component_class=OCRProcessorComponent,
        checkpoint_name="ckpt1_ocr_processed",
        checkpoint_number=1,
        description="Process files with OCR"
    )

model_config = RunnableComponentConfig(
    component_class=ModelManager,
    checkpoint_name="ckpt2_model_processing",
    checkpoint_number=2,
    description="Call to the model",
    component_configs=[
        ocr_processor_config
    ]
)
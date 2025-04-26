from typing import NamedTuple, Type, List, Optional, Dict, Any
from utils.runnable_component import RunnableComponent

class RunnableComponentConfig(NamedTuple):
    """
    Configuration for a pipeline component.
    Can be used with any class that inherits from RunnableComponent.
    
    Examples:
        # With Managers
        config = RunnableComponentConfig(
            component_class=ProcessingManager,
            checkpoint_name="processing",
            checkpoint_number=1,
            description="Process data files",
            component_configs=None  # Optional sub-components
        )
        
        # With Processors
        config = RunnableComponentConfig(
            component_class=OCRProcessor,
            checkpoint_name="ocr",
            checkpoint_number=1,
            description="Extract text from images",
            metadata={"supported_formats": ["png", "jpg"]}
        )
        
        # In pipeline configuration with sub-components
        pipeline_configs = [
            RunnableComponentConfig(
                component_class=SetupProcessor,
                checkpoint_name="setup",
                checkpoint_number=1,
                component_configs=[  # List of sub-components
                    RunnableComponentConfig(
                        component_class=ValidationProcessor,
                        checkpoint_name="validation",
                        checkpoint_number=1
                    )
                ]
            )
        ]
    """
    component_class: Type[RunnableComponent]
    checkpoint_name: str
    checkpoint_number: int
    description: str = ""  # Optional description of what this component does
    metadata: Dict[str, Any] = {}  # Additional metadata for validation/configuration
    component_configs: Optional[List['RunnableComponentConfig']] = None  # Optional list of sub-components
    